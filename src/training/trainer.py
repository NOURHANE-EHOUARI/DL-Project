"""
src/training/trainer.py — MTL Training Loop for Arabic NLP
===========================================================================
Owner  : Both (Student A + Student B)
Phase  : 3 — Modeling & Training (Week 3–5)

Responsibilities
----------------
  - Task sampling (proportional / uniform / round-robin via MTLBatchSampler)
  - Gradient accumulation for large effective batch sizes on limited GPU
  - Mixed precision training (torch.cuda.amp) for speed + memory efficiency
  - Warmup + cosine LR schedule (schedulers.py)
  - Uncertainty loss weighting (loss_weighting.py)
  - Per-epoch evaluation on NER F1, POS accuracy, Coref CoNLL-F1
  - W&B logging: losses, metrics, LR, gradient norms, loss weights
  - Checkpointing: best model per metric + periodic saves
  - Early stopping with configurable patience
  - Backbone freeze warm-up (N steps backbone frozen, then unfrozen)

Usage
-----
    from src.training.trainer import MTLTrainer, TrainerConfig

    config = TrainerConfig(
        backbone_name="aubmindlab/bert-base-arabertv2",
        num_epochs=10,
        batch_size=32,
        backbone_lr=2e-5,
        head_lr=4e-5,
        warmup_ratio=0.06,
        gradient_accumulation_steps=2,
        fp16=True,
        task_sampling_strategy="proportional",
        checkpoint_dir="checkpoints",
        wandb_project="arabic-mtl",
    )

    trainer = MTLTrainer(
        config=config,
        model=model,
        ner_train_loader=ner_train_loader,
        pos_train_loader=pos_train_loader,
        coref_train_loader=coref_train_loader,
        ner_eval_loader=ner_eval_loader,
        pos_eval_loader=pos_eval_loader,
        coref_eval_loader=coref_eval_loader,
    )
    trainer.train()
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning("wandb not installed — W&B logging disabled.")

try:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_AVAILABLE = True
except ImportError:
    _AMP_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Trainer Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainerConfig:
    """
    All hyperparameters and training settings in one place.
    Serializable to / from dict for W&B config logging.

    Fields
    ------
    backbone_name:
        HuggingFace model identifier for the Arabic backbone.
    num_epochs:
        Total training epochs.
    batch_size:
        Per-device batch size (before gradient accumulation).
    backbone_lr:
        Learning rate for the pre-trained backbone parameters.
    head_lr:
        Learning rate for randomly-initialized task head parameters.
    weight_decay:
        AdamW weight decay (applied to non-bias, non-LayerNorm params).
    warmup_ratio:
        Fraction of total steps used for linear LR warmup.
    min_lr_ratio:
        LR floor as fraction of peak LR (cosine decay floor).
    gradient_accumulation_steps:
        Accumulate gradients over N steps before optimizer.step().
        Effective batch size = batch_size × gradient_accumulation_steps.
    max_grad_norm:
        Gradient clipping threshold. 1.0 is standard for BERT fine-tuning.
    fp16:
        Enable automatic mixed precision (requires CUDA).
    task_sampling_strategy:
        'proportional' | 'uniform' | 'round_robin'
    checkpoint_dir:
        Directory for saving model checkpoints.
    save_every_n_steps:
        Save a periodic checkpoint every N optimizer steps (0 = disable).
    early_stopping_patience:
        Stop if NER F1 does not improve for this many epochs (0 = disable).
    freeze_backbone_steps:
        Keep backbone frozen for the first N optimizer steps.
    loss_weighting:
        'uncertainty' (Kendall) | 'fixed'
    fixed_weights:
        Task loss weights when loss_weighting='fixed'.
    wandb_project:
        W&B project name (empty string disables W&B).
    wandb_run_name:
        W&B run name (auto-generated if empty).
    log_every_n_steps:
        Log training metrics to W&B every N steps.
    eval_every_n_epochs:
        Run full evaluation every N epochs.
    seed:
        Random seed for reproducibility.
    """
    # Model
    backbone_name:          str   = "aubmindlab/bert-base-arabertv2"
    # Training
    num_epochs:             int   = 10
    batch_size:             int   = 32
    backbone_lr:            float = 2e-5
    head_lr:                float = 4e-5
    weight_decay:           float = 0.01
    warmup_ratio:           float = 0.06
    min_lr_ratio:           float = 0.05
    gradient_accumulation_steps: int = 1
    max_grad_norm:          float = 1.0
    fp16:                   bool  = True
    # Task sampling
    task_sampling_strategy: str   = "proportional"
    # Checkpointing
    checkpoint_dir:         str   = "checkpoints"
    save_every_n_steps:     int   = 500
    early_stopping_patience: int  = 3
    # Backbone freeze warm-up
    freeze_backbone_steps:  int   = 0
    # Loss weighting
    loss_weighting:         str   = "uncertainty"
    fixed_weights:          dict  = field(
        default_factory=lambda: {"ner": 1.0, "pos": 1.0, "coref": 1.0}
    )
    # Logging
    wandb_project:          str   = "arabic-mtl"
    wandb_run_name:         str   = ""
    log_every_n_steps:      int   = 50
    eval_every_n_epochs:    int   = 1
    seed:                   int   = 42

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Training metrics tracker
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpochMetrics:
    """Aggregated metrics for one training epoch."""
    epoch:          int
    ner_loss:       float = 0.0
    pos_loss:       float = 0.0
    coref_loss:     float = 0.0
    total_loss:     float = 0.0
    ner_f1:         float = 0.0
    pos_accuracy:   float = 0.0
    coref_f1:       float = 0.0
    duration_sec:   float = 0.0
    steps:          int   = 0

    def primary_metric(self) -> float:
        """NER F1 is the primary metric for early stopping and checkpointing."""
        return self.ner_f1

    def to_dict(self) -> dict[str, float]:
        return {
            "epoch":        self.epoch,
            "ner_loss":     self.ner_loss,
            "pos_loss":     self.pos_loss,
            "coref_loss":   self.coref_loss,
            "total_loss":   self.total_loss,
            "ner_f1":       self.ner_f1,
            "pos_accuracy": self.pos_accuracy,
            "coref_f1":     self.coref_f1,
            "duration_sec": self.duration_sec,
        }

    def summary(self) -> str:
        return (
            f"Epoch {self.epoch:3d} | "
            f"loss={self.total_loss:.4f} "
            f"(ner={self.ner_loss:.3f} pos={self.pos_loss:.3f} coref={self.coref_loss:.3f}) | "
            f"NER F1={self.ner_f1:.4f}  POS acc={self.pos_accuracy:.4f}  "
            f"Coref F1={self.coref_f1:.4f} | "
            f"{self.duration_sec:.0f}s"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MTL Trainer
# ══════════════════════════════════════════════════════════════════════════════

class MTLTrainer:
    """
    Custom training loop for the Arabic Multi-Task Learning model.

    Handles:
      - Task interleaving via MTLBatchSampler
      - Gradient accumulation + mixed precision
      - Warmup + cosine LR schedule
      - Per-task and total loss logging
      - NER F1, POS accuracy, Coref F1 evaluation
      - Best-model checkpointing + early stopping
      - Full W&B integration

    Args:
        config:             TrainerConfig with all hyperparameters.
        model:              ArabicMTLModel instance.
        ner_train_loader:   DataLoader for NER training data.
        pos_train_loader:   DataLoader for POS training data.
        coref_train_loader: DataLoader for Coref training data.
        ner_eval_loader:    DataLoader for NER evaluation data.
        pos_eval_loader:    DataLoader for POS evaluation data.
        coref_eval_loader:  DataLoader for Coref evaluation data.
        tokenizer:          Tokenizer (used for word_ids extraction).
        device:             Training device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        config:             TrainerConfig,
        model:              "ArabicMTLModel",    # forward ref — avoid circular
        ner_train_loader:   DataLoader,
        pos_train_loader:   DataLoader,
        coref_train_loader: DataLoader,
        ner_eval_loader:    Optional[DataLoader] = None,
        pos_eval_loader:    Optional[DataLoader] = None,
        coref_eval_loader:  Optional[DataLoader] = None,
        tokenizer:          Optional[Any] = None,
        device:             Optional[str] = None,
    ) -> None:
        self.config    = config
        self.model     = model
        self.tokenizer = tokenizer

        # Device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("Training on: %s", self.device)

        # Data loaders
        self.train_loaders = {
            "ner":   ner_train_loader,
            "pos":   pos_train_loader,
            "coref": coref_train_loader,
        }
        self.eval_loaders = {
            "ner":   ner_eval_loader,
            "pos":   pos_eval_loader,
            "coref": coref_eval_loader,
        }

        # Dataset sizes for MTLBatchSampler
        self.dataset_sizes = {
            task: len(loader.dataset)
            for task, loader in self.train_loaders.items()
        }

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Compute total steps
        from src.training.schedulers import compute_total_steps, compute_warmup_steps
        total_samples = sum(self.dataset_sizes.values())
        self.total_steps = compute_total_steps(
            num_train_samples=total_samples,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        self.warmup_steps = compute_warmup_steps(self.total_steps, config.warmup_ratio)
        logger.info(
            "Total optimizer steps: %d  Warmup: %d",
            self.total_steps, self.warmup_steps,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Mixed precision scaler
        self.scaler: Optional[GradScaler] = None
        if config.fp16 and _AMP_AVAILABLE and self.device == "cuda":
            self.scaler = GradScaler()
            logger.info("Mixed precision (FP16) enabled.")
        elif config.fp16:
            logger.warning("FP16 requested but CUDA/AMP unavailable. Using FP32.")

        # State
        self.global_step    = 0
        self.best_ner_f1    = 0.0
        self.no_improve     = 0
        self.history:  list[EpochMetrics] = []

        # Checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Reproducibility
        self._set_seed(config.seed)

        # W&B
        self._wandb_run = None
        if config.wandb_project and _WANDB_AVAILABLE:
            self._init_wandb()

    # ── Build optimizer and scheduler ────────────────────────────────────

    def _build_optimizer(self) -> AdamW:
        param_groups = self.model.get_optimizer_param_groups(
            backbone_lr=self.config.backbone_lr,
            head_lr=self.config.head_lr,
            weight_decay=self.config.weight_decay,
        )
        return AdamW(param_groups, eps=1e-8)

    def _build_scheduler(self):
        from src.training.schedulers import MTLScheduler
        return MTLScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.config.min_lr_ratio,
            head_lr_multipliers={
                "NERHead":        2.0,
                "POSHead":        2.0,
                "CoreferenceHead": 3.0,
            },
        )

    # ── Main training loop ────────────────────────────────────────────────

    def train(self) -> list[EpochMetrics]:
        """
        Run the full MTL training loop.

        Returns:
            List of EpochMetrics, one per epoch.
        """
        logger.info("Starting MTL training — %d epochs", self.config.num_epochs)
        if hasattr(self.model, "summary"):
            logger.info("\n%s", self.model.summary())

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()
            metrics = self._train_epoch(epoch)
            metrics.duration_sec = time.time() - t0

            # Evaluation
            if epoch % self.config.eval_every_n_epochs == 0:
                eval_metrics = self._evaluate(epoch)
                metrics.ner_f1       = eval_metrics.get("ner_f1",   0.0)
                metrics.pos_accuracy = eval_metrics.get("pos_acc",  0.0)
                metrics.coref_f1     = eval_metrics.get("coref_f1", 0.0)

            logger.info(metrics.summary())
            self.history.append(metrics)

            # W&B epoch log
            if self._wandb_run:
                wandb.log(metrics.to_dict(), step=self.global_step)

            # Checkpointing
            self._checkpoint(metrics, epoch)

            # Early stopping
            if self.config.early_stopping_patience > 0:
                if self.no_improve >= self.config.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d "
                        "(no NER F1 improvement for %d epochs)",
                        epoch, self.config.early_stopping_patience,
                    )
                    break

        logger.info(
            "Training complete. Best NER F1: %.4f", self.best_ner_f1
        )
        if self._wandb_run:
            wandb.finish()

        return self.history

    # ── Single epoch ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> EpochMetrics:
        """Run one epoch of MTL training."""
        self.model.train()
        metrics = EpochMetrics(epoch=epoch)

        task_loss_sums: dict[str, float] = {"ner": 0., "pos": 0., "coref": 0.}
        task_counts:    dict[str, int]   = {"ner": 0,  "pos": 0,  "coref": 0}

        # Task iterators — cycle infinitely, task sampler controls order
        iterators: dict[str, Iterator] = {
            task: iter(loader)
            for task, loader in self.train_loaders.items()
        }

        # Task sampling schedule for this epoch
        task_sequence = self._sample_tasks_for_epoch()
        accum_step    = 0
        self.optimizer.zero_grad()

        for step_idx, task in enumerate(task_sequence):
            # Get next batch for this task
            batch = self._next_batch(iterators, task)
            if batch is None:
                continue

            # Forward + loss
            loss, raw_loss = self._forward_step(batch, task)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Track losses
            task_loss_sums[task] += raw_loss
            task_counts[task]    += 1
            metrics.total_loss   += raw_loss
            metrics.steps        += 1
            accum_step           += 1

            # Optimizer step (every gradient_accumulation_steps)
            if accum_step % self.config.gradient_accumulation_steps == 0:
                self._optimizer_step()
                accum_step = 0

                # Step-level logging
                if (self.global_step % self.config.log_every_n_steps == 0
                        and self._wandb_run):
                    self._log_step(task, raw_loss, task_loss_sums, task_counts)

                # Periodic checkpoint
                if (self.config.save_every_n_steps > 0
                        and self.global_step % self.config.save_every_n_steps == 0):
                    self._save_periodic_checkpoint()

        # Compute average losses
        total_count = sum(task_counts.values())
        if total_count > 0:
            metrics.total_loss /= total_count
        metrics.ner_loss   = task_loss_sums["ner"]   / max(task_counts["ner"],   1)
        metrics.pos_loss   = task_loss_sums["pos"]   / max(task_counts["pos"],   1)
        metrics.coref_loss = task_loss_sums["coref"] / max(task_counts["coref"], 1)

        return metrics

    # ── Forward step ──────────────────────────────────────────────────────

    def _forward_step(
        self,
        batch: dict[str, Any],
        task:  str,
    ) -> tuple[torch.Tensor, float]:
        """
        Run one forward pass and return (weighted_loss, raw_loss_float).
        Handles FP16 via autocast if enabled.
        """
        # Move tensors to device
        batch = self._batch_to_device(batch)

        # Extract common fields
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch.get("token_type_ids")
        word_ids       = batch.get("word_ids")        # list — already on CPU

        # Task-specific labels
        ner_labels     = batch.get("ner_labels")
        pos_labels     = batch.get("pos_labels")
        coref_clusters = batch.get("clusters")
        morph_features = batch.get("morph_features")

        ctx = autocast() if (self.scaler is not None and _AMP_AVAILABLE) else _null_context()

        with ctx:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
                token_type_ids=token_type_ids,
                word_ids=word_ids,
                ner_labels=ner_labels,
                pos_labels=pos_labels,
                coref_clusters=coref_clusters,
                morph_features=morph_features,
            )

        loss     = outputs["loss"]
        raw_loss = outputs.get("raw_loss", loss).item()
        return loss, raw_loss

    # ── Optimizer step ────────────────────────────────────────────────────

    def _optimizer_step(self) -> None:
        """Clip gradients, step optimizer + scheduler, zero gradients."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        # Log gradient norm
        if (self._wandb_run
                and self.global_step % self.config.log_every_n_steps == 0):
            wandb.log({
                "grad_norm": grad_norm.item(),
                "lr/backbone": self.optimizer.param_groups[0]["lr"],
                "lr/head":     self.optimizer.param_groups[-1]["lr"],
            }, step=self.global_step)

    # ── Evaluation ────────────────────────────────────────────────────────

    def _evaluate(self, epoch: int) -> dict[str, float]:
        """
        Run evaluation on all tasks and return metrics dict.

        Returns:
            {
                "ner_f1":   float,
                "pos_acc":  float,
                "coref_f1": float,
            }
        """
        self.model.eval()
        results: dict[str, float] = {}

        # ── NER ──────────────────────────────────────────────────────────
        if self.eval_loaders.get("ner"):
            ner_f1 = self._eval_ner(self.eval_loaders["ner"])
            results["ner_f1"] = ner_f1
            logger.info("  Epoch %d  NER F1 = %.4f", epoch, ner_f1)

        # ── POS ──────────────────────────────────────────────────────────
        if self.eval_loaders.get("pos"):
            pos_acc = self._eval_pos(self.eval_loaders["pos"])
            results["pos_acc"] = pos_acc
            logger.info("  Epoch %d  POS Acc = %.4f", epoch, pos_acc)

        # ── Coref ─────────────────────────────────────────────────────────
        if self.eval_loaders.get("coref"):
            coref_f1 = self._eval_coref(self.eval_loaders["coref"])
            results["coref_f1"] = coref_f1
            logger.info("  Epoch %d  Coref F1 = %.4f", epoch, coref_f1)

        if self._wandb_run:
            wandb.log({f"eval/{k}": v for k, v in results.items()},
                      step=self.global_step)

        self.model.train()
        return results

    def _eval_ner(self, loader: DataLoader) -> float:
        """Entity-level F1 via NEREvaluator (seqeval)."""
        try:
            from src.evaluation.ner_eval import NEREvaluator
        except ImportError:
            logger.warning("NEREvaluator not available — returning 0.0")
            return 0.0

        all_gold, all_pred = [], []

        with torch.no_grad():
            for batch in loader:
                batch = self._batch_to_device(batch)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    task="ner",
                    token_type_ids=batch.get("token_type_ids"),
                    word_ids=batch.get("word_ids"),
                )
                preds = outputs["predictions"]   # (B, W)
                gold  = batch["ner_labels"]       # (B, W)

                B = preds.size(0)
                for b in range(B):
                    mask = gold[b] != -100
                    all_gold.append(gold[b][mask].cpu().tolist())
                    all_pred.append(preds[b][mask].cpu().tolist())

        if not all_gold:
            return 0.0

        # Convert IDs → tag strings for seqeval
        try:
            from src.models.ner_head import ID2LABEL as NER_ID2LABEL
        except ImportError:
            return 0.0

        gold_tags = [[NER_ID2LABEL.get(i, "O") for i in seq] for seq in all_gold]
        pred_tags = [[NER_ID2LABEL.get(i, "O") for i in seq] for seq in all_pred]

        evaluator = NEREvaluator(gold_seqs=gold_tags, pred_seqs=pred_tags)
        result    = evaluator.evaluate()
        return result.entity_f1

    def _eval_pos(self, loader: DataLoader) -> float:
        """Token-level POS accuracy."""
        correct = total = 0

        with torch.no_grad():
            for batch in loader:
                batch   = self._batch_to_device(batch)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    task="pos",
                    token_type_ids=batch.get("token_type_ids"),
                    word_ids=batch.get("word_ids"),
                )
                preds = outputs["predictions"]   # (B, W)
                gold  = batch["pos_labels"]       # (B, W)

                mask     = gold != -100
                correct += ((preds == gold) & mask).sum().item()
                total   += mask.sum().item()

        return correct / total if total > 0 else 0.0

    def _eval_coref(self, loader: DataLoader) -> float:
        """
        Simplified CoNLL-F1 approximation.
        Full CoNLL scorer requires Student B's coref_eval.py.
        Returns MUC F1 approximation based on cluster overlap.
        """
        total_f1 = 0.0
        count    = 0

        with torch.no_grad():
            for batch in loader:
                batch   = self._batch_to_device(batch)
                gold_clusters = batch.get("clusters", [])
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    task="coref",
                    token_type_ids=batch.get("token_type_ids"),
                    coref_clusters=gold_clusters,
                )
                pred_clusters = outputs["predictions"]

                for b in range(len(gold_clusters)):
                    gold = gold_clusters[b] if b < len(gold_clusters) else []
                    pred = pred_clusters[b] if b < len(pred_clusters) else []
                    f1   = self._muc_f1_approx(gold, pred)
                    total_f1 += f1
                    count    += 1

        return total_f1 / count if count > 0 else 0.0

    @staticmethod
    def _muc_f1_approx(
        gold_clusters: list,
        pred_clusters: list,
    ) -> float:
        """
        Approximate MUC F1: compare sets of mention pairs.
        Full CoNLL scorer is in src/evaluation/coref_eval.py (Student B).
        """
        def cluster_to_pairs(clusters):
            pairs = set()
            for cluster in clusters:
                mentions = [tuple(m) for m in cluster]
                for i in range(len(mentions)):
                    for j in range(i + 1, len(mentions)):
                        pairs.add((mentions[i], mentions[j]))
            return pairs

        gold_pairs = cluster_to_pairs(gold_clusters)
        pred_pairs = cluster_to_pairs(pred_clusters)

        if not gold_pairs and not pred_pairs:
            return 1.0
        if not gold_pairs or not pred_pairs:
            return 0.0

        tp = len(gold_pairs & pred_pairs)
        p  = tp / len(pred_pairs)
        r  = tp / len(gold_pairs)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # ── Task sampling ─────────────────────────────────────────────────────

    def _sample_tasks_for_epoch(self) -> list[str]:
        """
        Generate the task sequence for one epoch using MTLBatchSampler.
        """
        from src.data.collators import MTLBatchSampler
        sampler = MTLBatchSampler(
            dataset_sizes=self.dataset_sizes,
            batch_size=self.config.batch_size,
            strategy=self.config.task_sampling_strategy,
            seed=self.config.seed + self.global_step,
        )
        return list(sampler)

    def _next_batch(
        self,
        iterators: dict[str, Iterator],
        task: str,
    ) -> Optional[dict[str, Any]]:
        """Get next batch from task iterator, reinitializing if exhausted."""
        try:
            return next(iterators[task])
        except StopIteration:
            iterators[task] = iter(self.train_loaders[task])
            try:
                return next(iterators[task])
            except StopIteration:
                return None

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _checkpoint(self, metrics: EpochMetrics, epoch: int) -> None:
        """Save best model checkpoint and update early stopping counter."""
        ner_f1 = metrics.ner_f1

        if ner_f1 > self.best_ner_f1:
            self.best_ner_f1 = ner_f1
            self.no_improve  = 0
            path = os.path.join(self.config.checkpoint_dir, "best_mtl.pt")
            self.model.save_checkpoint(
                path,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics=metrics.to_dict(),
            )
            logger.info(
                "  ✅ New best NER F1=%.4f — checkpoint saved to %s",
                ner_f1, path,
            )
        else:
            self.no_improve += 1
            logger.info(
                "  No improvement (%d/%d)",
                self.no_improve, self.config.early_stopping_patience,
            )

    def _save_periodic_checkpoint(self) -> None:
        """Save a checkpoint keyed by global step."""
        path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_step_{self.global_step}.pt",
        )
        self.model.save_checkpoint(
            path, optimizer=self.optimizer, scheduler=self.scheduler,
            metrics={"global_step": self.global_step},
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    def _batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move all tensor values in a batch dict to self.device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_step(
        self,
        task:            str,
        raw_loss:        float,
        task_loss_sums:  dict[str, float],
        task_counts:     dict[str, int],
    ) -> None:
        """Log per-step metrics to W&B."""
        if not self._wandb_run:
            return
        log_dict: dict[str, Any] = {
            f"train/loss_{task}": raw_loss,
            "train/global_step":  self.global_step,
        }
        # Log uncertainty weights if available
        if hasattr(self.model.loss_weighter, "log_vars"):
            for i, t in enumerate(("ner", "pos", "coref")):
                w = torch.exp(-self.model.loss_weighter.log_vars[i]).item()
                log_dict[f"loss_weight/{t}"] = w
        wandb.log(log_dict, step=self.global_step)

    def _init_wandb(self) -> None:
        """Initialize W&B run."""
        try:
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or None,
                config=self.config.to_dict(),
                resume="allow",
            )
            logger.info("W&B run initialized: %s", self._wandb_run.url)
        except Exception as exc:
            logger.warning("W&B init failed: %s", exc)
            self._wandb_run = None

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ── Context manager helper ────────────────────────────────────────────────────

class _null_context:
    """No-op context manager (fallback when AMP is unavailable)."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test (no model download — pure config + logic validation)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n── TrainerConfig smoke test ────────────────────────────")

    cfg = TrainerConfig(
        backbone_name="aubmindlab/bert-base-arabertv2",
        num_epochs=10,
        batch_size=32,
        backbone_lr=2e-5,
        head_lr=4e-5,
        warmup_ratio=0.06,
        gradient_accumulation_steps=2,
        fp16=True,
        task_sampling_strategy="proportional",
        checkpoint_dir="checkpoints",
        wandb_project="arabic-mtl",
    )
    d = cfg.to_dict()
    assert d["num_epochs"] == 10
    assert d["backbone_lr"] == 2e-5
    print("  ✅ TrainerConfig.to_dict() OK")

    print("\n── EpochMetrics smoke test ─────────────────────────────")
    m = EpochMetrics(epoch=1, ner_loss=0.4, pos_loss=0.2, coref_loss=1.1,
                     total_loss=0.57, ner_f1=0.832, pos_accuracy=0.961,
                     coref_f1=0.618, duration_sec=143.0, steps=500)
    print(f"  {m.summary()}")
    assert m.primary_metric() == 0.832
    print("  ✅ EpochMetrics OK")

    print("\n── MUC F1 approximation test ───────────────────────────")
    gold = [[[[0,1],[5,6]],[[10,10],[15,16]]]]
    pred = [[[[0,1],[5,6]],[[10,10],[14,16]]]]   # one mention off
    f1 = MTLTrainer._muc_f1_approx(gold[0], pred[0])
    assert 0.0 < f1 < 1.0, f"Expected partial F1, got {f1}"
    print(f"  ✅ MUC F1 (partial match) = {f1:.4f}")

    perfect_f1 = MTLTrainer._muc_f1_approx(gold[0], gold[0])
    assert perfect_f1 == 1.0
    print(f"  ✅ MUC F1 (perfect match) = {perfect_f1:.4f}")

    print("\n── trainer.py ready ✅\n")