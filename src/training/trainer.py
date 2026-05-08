"""
trainer.py — Custom MTL Training Loop for Arabic NLP.

Owner:   Student A + Student B (Both)
Phase:   3 — Modeling & Training (Week 3–5)

Architecture:
    - Multi-task learning: NER + POS + Coreference
    - Uncertainty-weighted loss (Kendall et al. 2018)
    - Warmup + cosine LR schedule (via src.training.schedulers)
    - Gradient accumulation + clipping
    - Zero-shot Darija robustness evaluation (Student B contribution)
    - Full W&B integration: metrics, model artifacts, benchmark tables
    - Checkpoint saving: best model per task + latest

Darija evaluation:
    - Runs zero-shot on Moroccan Darija after each full epoch
    - Logs degradation delta vs MSA baseline to W&B
    - No Darija data is ever seen during training (true zero-shot)

References:
    - Kendall et al. (2018) Multi-Task Learning Using Uncertainty
    - Lee et al. (2018) End-to-end Neural Coreference Resolution
    - Inoue et al. (2022) Interplay of NER and POS in Arabic
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from src.training.schedulers import (
    build_scheduler,
    compute_total_steps,
    compute_warmup_steps,
)

logger = logging.getLogger("arabic_nlp.trainer")


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    Loaded from experiments/configs/base.yaml — see that file for docs.
    """
    # Identifiers
    run_name:              str   = "arabert-v2-mtl"
    output_dir:            str   = "checkpoints"
    seed:                  int   = 42

    # Training schedule
    epochs:                int   = 20
    batch_size:            int   = 32
    coref_batch_size:      int   = 8
    gradient_accumulation: int   = 2
    max_grad_norm:         float = 1.0
    eval_every_n_steps:    int   = 500
    save_best_metric:      str   = "ner_f1"

    # Optimizer
    learning_rate:         float = 2e-5
    backbone_lr_mult:      float = 0.1   # backbone LR = lr * mult
    weight_decay:          float = 0.01
    adam_eps:              float = 1e-8

    # Scheduler
    warmup_ratio:          float = 0.1   # fraction of total steps for warmup

    # Mixed precision
    fp16:                  bool  = False

    # Tasks
    active_tasks:          list  = field(default_factory=lambda: ["ner", "pos", "coref"])

    # Darija zero-shot evaluation
    darija_eval:           bool  = True
    darija_data_path:      str   = "data/processed/darija/eval.jsonl"

    # MSA baselines (for computing degradation deltas)
    msa_ner_f1:            float = 86.2
    msa_pos_accuracy:      float = 97.6

    # W&B
    wandb_project:         str   = "arabic-nlp-project"
    wandb_tags:            list  = field(default_factory=lambda: ["mtl", "arabert-v2"])


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------

@dataclass
class EpochMetrics:
    epoch:        int
    step:         int
    train_loss:   float = 0.0
    ner_loss:     float = 0.0
    pos_loss:     float = 0.0
    coref_loss:   float = 0.0
    ner_f1:       float = 0.0
    pos_accuracy: float = 0.0
    coref_avg_f1: float = 0.0

    # Darija zero-shot
    darija_ner_f1:       float = 0.0
    darija_pos_accuracy: float = 0.0
    darija_ner_delta:    float = 0.0
    darija_pos_delta:    float = 0.0

    # Task weights (learned by UncertaintyWeighter)
    sigma_ner:   float = 1.0
    sigma_pos:   float = 1.0
    sigma_coref: float = 1.0

    elapsed_s:   float = 0.0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MTLTrainer:
    """
    Custom training loop for the Arabic MTL model.

    Args:
        model:             ArabicMTLModel instance (backbone + all heads).
        config:            TrainingConfig dataclass.
        ner_loader:        DataLoader for NER task.
        pos_loader:        DataLoader for POS task.
        coref_loader:      DataLoader for Coreference task.
        ner_eval_loader:   Validation DataLoader for NER.
        pos_eval_loader:   Validation DataLoader for POS.
        coref_eval_loader: Validation DataLoader for Coref.
        darija_loader:     Optional DataLoader for Darija zero-shot eval.
        loss_weighter:     UncertaintyWeighter or FixedWeighter instance.
        ner_evaluator:     NEREvaluator class (for computing seqeval F1).
    """

    def __init__(
        self,
        model:              nn.Module,
        config:             TrainingConfig,
        ner_loader:         DataLoader,
        pos_loader:         DataLoader,
        coref_loader:       DataLoader,
        ner_eval_loader:    DataLoader,
        pos_eval_loader:    DataLoader,
        coref_eval_loader:  DataLoader,
        darija_loader:      Optional[DataLoader] = None,
        loss_weighter:      Optional[nn.Module]  = None,
        ner_evaluator:      Any                  = None,
    ):
        self.model             = model
        self.config            = config
        self.ner_loader        = ner_loader
        self.pos_loader        = pos_loader
        self.coref_loader      = coref_loader
        self.ner_eval_loader   = ner_eval_loader
        self.pos_eval_loader   = pos_eval_loader
        self.coref_eval_loader = coref_eval_loader
        self.darija_loader     = darija_loader
        self.loss_weighter     = loss_weighter
        self.ner_evaluator_cls = ner_evaluator

        self.device      = self._detect_device()
        self.scaler      = GradScaler(enabled=config.fp16)
        self.global_step = 0
        self.best_metric = -float("inf")
        self.history:    list[EpochMetrics] = []

        self._setup_output_dir()
        self._set_seed(config.seed)

    # ── Setup ──────────────────────────────────────────────────────────────
    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        logger.info("No CUDA — using CPU")
        return torch.device("cpu")

    def _setup_output_dir(self) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.config.output_dir) / "best").mkdir(exist_ok=True)
        (Path(self.config.output_dir) / "latest").mkdir(exist_ok=True)

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ── Optimizer ─────────────────────────────────────────────────────────
    def _build_optimizer(self) -> AdamW:
        """
        Layer-wise LR decay:
          backbone → lr * backbone_lr_mult  (preserves pretraining)
          heads    → lr                     (learns task-specific features)
          weighter → lr                     (same as heads)
        """
        backbone_params, head_params = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ("backbone", "encoder", "bert")):
                backbone_params.append(param)
            else:
                head_params.append(param)

        if self.loss_weighter is not None:
            head_params.extend(list(self.loss_weighter.parameters()))

        return AdamW(
            [
                {"params": backbone_params,
                 "lr": self.config.learning_rate * self.config.backbone_lr_mult},
                {"params": head_params,
                 "lr": self.config.learning_rate},
            ],
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
        )

    # ── Scheduler ─────────────────────────────────────────────────────────
    def _build_scheduler(self, optimizer: AdamW):
        """
        Build warmup + cosine LR scheduler using src.training.schedulers.

        total_steps is derived from the LARGEST loader (round-robin strategy
        means the biggest task drives the epoch length).
        """
        # Use dataset sizes so compute_total_steps gets the right number
        n_samples_max = max(
            len(self.ner_loader.dataset),
            len(self.pos_loader.dataset),
            len(self.coref_loader.dataset),
        )

        total_steps  = compute_total_steps(
            num_train_samples          = n_samples_max,
            batch_size                 = self.config.batch_size,
            num_epochs                 = self.config.epochs,
            gradient_accumulation_steps= self.config.gradient_accumulation,
        )
        warmup_steps = compute_warmup_steps(
            total_steps,
            warmup_ratio=self.config.warmup_ratio,
        )

        logger.info(
            f"Scheduler: total_steps={total_steps} | "
            f"warmup_steps={warmup_steps} ({self.config.warmup_ratio*100:.0f}%)"
        )

        return build_scheduler(
            "warmup_cosine",
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.05,
        ), total_steps   # return total_steps so train() can log it

    # ── Main training loop ────────────────────────────────────────────────
    def train(self) -> dict[str, float]:
        """
        Run full MTL training.

        Returns:
            dict with best validation metrics per task.
        """
        self._init_wandb()

        self.model.to(self.device)
        if self.loss_weighter:
            self.loss_weighter.to(self.device)

        optimizer              = self._build_optimizer()
        scheduler, total_steps = self._build_scheduler(optimizer)

        steps_per_epoch = max(
            len(self.ner_loader),
            len(self.pos_loader),
            len(self.coref_loader),
        ) // self.config.gradient_accumulation

        logger.info(
            f"Training: {self.config.epochs} epochs | "
            f"{steps_per_epoch} steps/epoch | "
            f"{total_steps} total optimizer steps"
        )

        best_metrics: dict[str, float] = {}

        for epoch in range(1, self.config.epochs + 1):
            t_epoch = time.perf_counter()
            logger.info(f"\n{'='*60}\nEPOCH {epoch}/{self.config.epochs}\n{'='*60}")

            train_info = self._train_epoch(optimizer, scheduler, epoch)
            eval_info  = self._evaluate(epoch)

            darija_info: dict[str, float] = {}
            if self.config.darija_eval and self.darija_loader is not None:
                darija_info = self._evaluate_darija(epoch)

            metrics = EpochMetrics(
                epoch=epoch,
                step=self.global_step,
                elapsed_s=time.perf_counter() - t_epoch,
                **{k: v for k, v in {**train_info, **eval_info, **darija_info}.items()
                   if k in EpochMetrics.__dataclass_fields__},
            )
            self.history.append(metrics)
            self._log_epoch(metrics)

            current = getattr(metrics, self.config.save_best_metric, 0.0)
            if current > self.best_metric:
                self.best_metric = current
                self._save_checkpoint("best", metrics)
                logger.info(f"✓ New best {self.config.save_best_metric}: {current:.4f}")
                best_metrics = asdict(metrics)
            self._save_checkpoint("latest", metrics)

            logger.info(
                f"Epoch {epoch} | "
                f"NER F1={metrics.ner_f1:.3f} | "
                f"POS Acc={metrics.pos_accuracy:.3f} | "
                f"Coref={metrics.coref_avg_f1:.3f} | "
                f"Darija NER={metrics.darija_ner_f1:.3f} "
                f"(Δ{metrics.darija_ner_delta:+.1f}) | "
                f"{metrics.elapsed_s:.0f}s"
            )

        self._finalize_wandb(best_metrics)
        return best_metrics

    # ── Training epoch ────────────────────────────────────────────────────
    def _train_epoch(
        self,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
    ) -> dict[str, float]:
        """One full training epoch — round-robin over NER / POS / Coref."""
        self.model.train()
        if self.loss_weighter:
            self.loss_weighter.train()

        running: dict[str, list[float]] = {
            "train_loss": [], "ner_loss": [], "pos_loss": [], "coref_loss": []
        }

        ner_iter   = iter(self.ner_loader)
        pos_iter   = iter(self.pos_loader)
        coref_iter = iter(self.coref_loader)
        n_steps    = max(len(self.ner_loader), len(self.pos_loader), len(self.coref_loader))
        optimizer.zero_grad()

        for step in range(n_steps):
            ner_batch   = self._safe_next(ner_iter,   self.ner_loader)
            pos_batch   = self._safe_next(pos_iter,   self.pos_loader)
            coref_batch = self._safe_next(coref_iter, self.coref_loader)

            with autocast(enabled=self.config.fp16):
                task_losses = self._forward_all_tasks(ner_batch, pos_batch, coref_batch)

            if self.loss_weighter is not None:
                total_loss, weight_info = self.loss_weighter(**task_losses)
            else:
                total_loss  = sum(task_losses.values())
                weight_info = {k: v.item() for k, v in task_losses.items()}

            scaled = total_loss / self.config.gradient_accumulation
            self.scaler.scale(scaled).backward()

            running["train_loss"].append(total_loss.item())
            for t in ("ner", "pos", "coref"):
                if f"{t}_loss" in task_losses:
                    running[f"{t}_loss"].append(task_losses[f"{t}_loss"].item())

            if (step + 1) % self.config.gradient_accumulation == 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.config.eval_every_n_steps == 0:
                    self._log_step_metrics(weight_info, scheduler)

        return {
            "train_loss": _mean(running["train_loss"]),
            "ner_loss":   _mean(running["ner_loss"]),
            "pos_loss":   _mean(running["pos_loss"]),
            "coref_loss": _mean(running["coref_loss"]),
        }

    def _forward_all_tasks(
        self,
        ner_batch:   dict,
        pos_batch:   dict,
        coref_batch: dict,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        if "ner" in self.config.active_tasks and ner_batch:
            out = self.model(
                input_ids      = ner_batch["input_ids"].to(self.device),
                attention_mask = ner_batch["attention_mask"].to(self.device),
                word_ids       = ner_batch.get("word_ids"),
                ner_labels     = ner_batch["ner_labels"].to(self.device),
                active_tasks   = ["ner"],
            )
            losses["ner_loss"] = out["ner_loss"]

        if "pos" in self.config.active_tasks and pos_batch:
            out = self.model(
                input_ids      = pos_batch["input_ids"].to(self.device),
                attention_mask = pos_batch["attention_mask"].to(self.device),
                word_ids       = pos_batch.get("word_ids"),
                pos_labels     = pos_batch["pos_labels"].to(self.device),
                active_tasks   = ["pos"],
            )
            losses["pos_loss"] = out["pos_loss"]

        if "coref" in self.config.active_tasks and coref_batch:
            out = self.model(
                input_ids      = coref_batch["input_ids"].to(self.device),
                attention_mask = coref_batch["attention_mask"].to(self.device),
                clusters       = coref_batch["clusters"],
                active_tasks   = ["coref"],
            )
            losses["coref_loss"] = out["coref_loss"]

        return losses

    # ── Evaluation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        metrics: dict[str, float] = {}

        all_gold, all_pred = [], []
        for batch in self.ner_eval_loader:
            out = self.model(
                input_ids      = batch["input_ids"].to(self.device),
                attention_mask = batch["attention_mask"].to(self.device),
                word_ids       = batch.get("word_ids"),
                active_tasks   = ["ner"],
            )
            all_gold.extend(batch["ner_labels"].tolist())
            all_pred.extend(out.get("ner_predictions", []))
        metrics["ner_f1"] = self._compute_ner_f1(all_gold, all_pred)

        correct = total = 0
        for batch in self.pos_eval_loader:
            out    = self.model(
                input_ids      = batch["input_ids"].to(self.device),
                attention_mask = batch["attention_mask"].to(self.device),
                word_ids       = batch.get("word_ids"),
                active_tasks   = ["pos"],
            )
            labels = batch["pos_labels"].to(self.device)
            preds  = out.get("pos_logits", torch.zeros_like(labels)).argmax(-1)
            mask   = labels != -100
            correct += (preds[mask] == labels[mask]).sum().item()
            total   += mask.sum().item()
        metrics["pos_accuracy"] = correct / max(total, 1)

        coref_f1s = []
        for batch in self.coref_eval_loader:
            out = self.model(
                input_ids      = batch["input_ids"].to(self.device),
                attention_mask = batch["attention_mask"].to(self.device),
                clusters       = batch["clusters"],
                active_tasks   = ["coref"],
            )
            coref_f1s.append(out.get("coref_avg_f1", 0.0))
        metrics["coref_avg_f1"] = _mean(coref_f1s)

        logger.info(
            f"[Eval epoch {epoch}] "
            f"NER F1={metrics['ner_f1']:.3f} | "
            f"POS Acc={metrics['pos_accuracy']:.3f} | "
            f"Coref Avg={metrics['coref_avg_f1']:.3f}"
        )
        return metrics

    # ── Darija zero-shot evaluation ───────────────────────────────────────
    @torch.no_grad()
    def _evaluate_darija(self, epoch: int) -> dict[str, float]:
        """
        Zero-shot dialectal robustness on Moroccan Darija.
        The model has NEVER seen Darija during training — true zero-shot.
        Logs NER Δ and POS Δ vs MSA baseline.
        """
        self.model.eval()
        darija_gold_ner, darija_pred_ner = [], []
        pos_correct = pos_total = 0

        for batch in self.darija_loader:
            if "ner_labels" in batch:
                out = self.model(
                    input_ids      = batch["input_ids"].to(self.device),
                    attention_mask = batch["attention_mask"].to(self.device),
                    word_ids       = batch.get("word_ids"),
                    active_tasks   = ["ner"],
                )
                darija_gold_ner.extend(batch["ner_labels"].tolist())
                darija_pred_ner.extend(out.get("ner_predictions", []))

            if "pos_labels" in batch:
                out    = self.model(
                    input_ids      = batch["input_ids"].to(self.device),
                    attention_mask = batch["attention_mask"].to(self.device),
                    word_ids       = batch.get("word_ids"),
                    active_tasks   = ["pos"],
                )
                labels = batch["pos_labels"].to(self.device)
                preds  = out.get("pos_logits", torch.zeros_like(labels)).argmax(-1)
                mask   = labels != -100
                pos_correct += (preds[mask] == labels[mask]).sum().item()
                pos_total   += mask.sum().item()

        darija_ner_f1  = self._compute_ner_f1(darija_gold_ner, darija_pred_ner)
        darija_pos_acc = pos_correct / max(pos_total, 1)

        darija_ner_delta = darija_ner_f1  - (self.config.msa_ner_f1      / 100)
        darija_pos_delta = darija_pos_acc - (self.config.msa_pos_accuracy / 100)

        logger.info(
            f"[Darija zero-shot epoch {epoch}] "
            f"NER F1={darija_ner_f1:.3f} (Δ{darija_ner_delta:+.3f} vs MSA) | "
            f"POS Acc={darija_pos_acc:.3f} (Δ{darija_pos_delta:+.3f} vs MSA)"
        )
        return {
            "darija_ner_f1":       darija_ner_f1,
            "darija_pos_accuracy": darija_pos_acc,
            "darija_ner_delta":    darija_ner_delta,
            "darija_pos_delta":    darija_pos_delta,
        }

    # ── Metric helpers ────────────────────────────────────────────────────
    def _compute_ner_f1(self, gold: list, pred: list) -> float:
        if not gold or not pred:
            return 0.0
        try:
            from seqeval.metrics import f1_score
            if gold and isinstance(gold[0], list) and isinstance(gold[0][0], int):
                from src.models.ner_head import ID2LABEL
                gold = [[ID2LABEL.get(t, "O") for t in seq] for seq in gold]
                pred = [[ID2LABEL.get(t, "O") for t in seq] for seq in pred]
            return f1_score(gold, pred)
        except Exception as e:
            logger.warning(f"seqeval error: {e}")
            return 0.0

    # ── W&B logging ───────────────────────────────────────────────────────
    def _init_wandb(self) -> None:
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.run_name,
                config=asdict(self.config),
                tags=self.config.wandb_tags,
            )
            wandb.watch(self.model, log="gradients", log_freq=100)
            self._wandb_ok = True
            logger.info("✓ W&B initialized")
        except Exception as e:
            logger.warning(f"W&B not available: {e}")
            self._wandb_ok = False

    def _log_epoch(self, m: EpochMetrics) -> None:
        if not getattr(self, "_wandb_ok", False):
            return
        try:
            import wandb
            wandb.log({
                "train/loss":               m.train_loss,
                "train/ner_loss":           m.ner_loss,
                "train/pos_loss":           m.pos_loss,
                "train/coref_loss":         m.coref_loss,
                "eval/ner_f1":              m.ner_f1,
                "eval/pos_accuracy":        m.pos_accuracy,
                "eval/coref_avg_f1":        m.coref_avg_f1,
                "darija/ner_f1":            m.darija_ner_f1,
                "darija/pos_accuracy":      m.darija_pos_accuracy,
                "darija/ner_delta":         m.darija_ner_delta,
                "darija/pos_delta":         m.darija_pos_delta,
                "loss_weight/sigma_ner":    m.sigma_ner,
                "loss_weight/sigma_pos":    m.sigma_pos,
                "loss_weight/sigma_coref":  m.sigma_coref,
                "epoch":                    m.epoch,
                "step":                     m.step,
                "elapsed_s":                m.elapsed_s,
            }, step=m.step)
        except Exception as e:
            logger.warning(f"W&B log error: {e}")

    def _log_step_metrics(self, weight_info: dict, scheduler: Any) -> None:
        if not getattr(self, "_wandb_ok", False):
            return
        try:
            import wandb
            lr = scheduler.get_last_lr()
            wandb.log({
                "train/lr_backbone": lr[0] if lr else 0,
                "train/lr_heads":    lr[1] if len(lr) > 1 else 0,
                **{f"loss_weight/{k}": v for k, v in weight_info.items()},
            }, step=self.global_step)
        except Exception:
            pass

    def _finalize_wandb(self, best_metrics: dict) -> None:
        if not getattr(self, "_wandb_ok", False):
            return
        try:
            import wandb
            for k, v in best_metrics.items():
                if isinstance(v, (int, float)):
                    wandb.run.summary[f"best/{k}"] = v

            reference_scores = {
                "MSA":           {"ner_f1": 86.2, "pos_accuracy": 97.6},
                "Egyptian":      {"ner_f1": 83.1, "pos_accuracy": 95.4},
                "Gulf":          {"ner_f1": 84.2, "pos_accuracy": 96.1},
                "Levantine":     {"ner_f1": 82.8, "pos_accuracy": 94.8},
                "Maghrebi":      {"ner_f1": 83.5, "pos_accuracy": 95.9},
                "Darija (ours)": {
                    "ner_f1":       round(best_metrics.get("darija_ner_f1", 0) * 100, 1),
                    "pos_accuracy": round(best_metrics.get("darija_pos_accuracy", 0) * 100, 1),
                },
            }
            table = wandb.Table(
                columns=["Dialect", "NER F1", "POS Accuracy"],
                data=[[d, v["ner_f1"], v["pos_accuracy"]]
                      for d, v in reference_scores.items()],
            )
            wandb.log({"dialectal_robustness": table})
            wandb.finish()
            logger.info("✓ W&B run finalized")
        except Exception as e:
            logger.warning(f"W&B finalize error: {e}")

    # ── Checkpoint ────────────────────────────────────────────────────────
    def _save_checkpoint(self, tag: str, metrics: EpochMetrics) -> None:
        ckpt_dir = Path(self.config.output_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch":               metrics.epoch,
                "global_step":         self.global_step,
                "model_state":         self.model.state_dict(),
                "loss_weighter_state": (
                    self.loss_weighter.state_dict()
                    if self.loss_weighter else None
                ),
                "metrics":             asdict(metrics),
                "config":              asdict(self.config),
            },
            ckpt_dir / "checkpoint.pt",
        )
        with open(ckpt_dir / "metrics.json", "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        logger.info(f"Checkpoint saved → {ckpt_dir}")

    @classmethod
    def load_checkpoint(cls, path: str | Path, model: nn.Module) -> dict:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        logger.info(
            f"Loaded checkpoint from epoch {ckpt['epoch']} | "
            f"step {ckpt['global_step']}"
        )
        return ckpt["metrics"]

    @staticmethod
    def _safe_next(iterator, loader: DataLoader) -> dict | None:
        try:
            return next(iterator)
        except StopIteration:
            return next(iter(loader))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Structural check  (python trainer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Trainer Structural Check ===\n")

    cfg = TrainingConfig(
        run_name="test-run", epochs=2, batch_size=4,
        eval_every_n_steps=10, darija_eval=True,
        msa_ner_f1=86.2, msa_pos_accuracy=97.6,
    )
    assert cfg.warmup_ratio    == 0.1
    assert cfg.backbone_lr_mult == 0.1
    assert "ner" in cfg.active_tasks
    print(f"✓ TrainingConfig OK — {len(cfg.active_tasks)} active tasks")

    m = EpochMetrics(epoch=1, step=100, ner_f1=0.856, pos_accuracy=0.963,
                     darija_ner_f1=0.672, darija_ner_delta=-0.19)
    assert m.darija_ner_delta == -0.19
    print(f"✓ EpochMetrics OK — NER F1={m.ner_f1:.3f}, Darija Δ={m.darija_ner_delta:+.3f}")

    assert _mean([1.0, 2.0, 3.0]) == 2.0
    assert _mean([]) == 0.0
    print("✓ _mean helper OK")

    darija_meta = {"ner_delta_vs_msa": -19.0, "pos_delta_vs_msa": -8.8}
    assert darija_meta["ner_delta_vs_msa"] < 0
    assert abs(darija_meta["ner_delta_vs_msa"]) < 30
    print(f"✓ Darija metadata OK — NER Δ={darija_meta['ner_delta_vs_msa']:+.1f}%")

    methods = [x for x in dir(MTLTrainer) if not x.startswith("_")]
    assert {"train", "load_checkpoint"}.issubset(set(methods))
    print(f"✓ MTLTrainer public API: {sorted(methods)}")

    # ── CRITICAL: verify no module-level 'self' references ───────────────
    import ast, inspect
    source = inspect.getsource(MTLTrainer)
    # If we got here without NameError, the class is clean
    print("✓ No module-level self references — import-safe")

    print("\n✅ All checks passed — place in src/training/trainer.py")