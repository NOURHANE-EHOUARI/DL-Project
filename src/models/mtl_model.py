"""
src/models/mtl_model.py — Unified Multi-Task Learning Model for Arabic NLP
===========================================================================
Owner  : Both (Student A + Student B)
Phase  : 3 — Modeling & Training (Week 3–5)

Architecture
------------
  Raw Arabic text
       ↓
  ArabicBackbone  (AraBERT v2 / CAMeLBERT / AraELECTRA)
  Shared transformer encoder — one forward pass for all tasks
       ↓
  ┌────────────────────────────────────────┐
  │  Task-specific heads (in parallel)     │
  │  ┌──────────┐ ┌──────────┐ ┌────────┐ │
  │  │ NERHead  │ │ POSHead  │ │ Coref  │ │
  │  │ BiLSTM   │ │ Token    │ │ Span   │ │
  │  │ -CRF     │ │ Classif. │ │ Scorer │ │
  │  └──────────┘ └──────────┘ └────────┘ │
  └────────────────────────────────────────┘
       ↓
  UncertaintyWeighter (Kendall et al. 2018)
  Learned task weights → weighted sum of losses

Design decisions
----------------
  1. Single backbone forward pass per batch — most efficient MTL design.
     Each batch contains ONE task's examples (task sampling in trainer).
     The backbone runs once, only the relevant head computes its loss.

  2. Shared BiLSTM adapter bridge — optional 1-layer BiLSTM on top of the
     transformer outputs before task heads. Improves cross-task transfer
     for sequence labeling tasks (NER + POS share the same bridge).

  3. Gradient isolation — coref head gets raw subword outputs; NER and POS
     heads receive word-level pooled outputs (first-subword strategy).
     This matches each head's optimal input representation.

  4. Uncertainty loss weighting (Kendall et al., CVPR 2018) — the model
     learns optimal per-task loss weights via homoscedastic uncertainty.
     Replaces manual weight tuning; cite this in the jury Q&A.

  5. Backbone freeze scheduling — the backbone can be frozen for the first
     N steps (head warm-up phase) then unfrozen for full fine-tuning.
     Prevents catastrophic forgetting early in training.

Usage
-----
    model = ArabicMTLModel.from_pretrained("aubmindlab/bert-base-arabertv2")

    # NER batch
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        task="ner",
        ner_labels=batch["ner_labels"],
    )
    loss = outputs["loss"]

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=..., attention_mask=..., task="ner")
    predictions = outputs["predictions"]
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

# ── Local imports ─────────────────────────────────────────────────────────────
# These are the heads built by Student A and Student B
from src.models.ner_head   import NERHead
from src.models.pos_head   import POSHead
from src.models.coref_head import CoreferenceHead
from src.training.loss_weighting import UncertaintyWeighter, FixedWeighter

logger = logging.getLogger(__name__)

# ── Supported task names ──────────────────────────────────────────────────────
TASKS = ("ner", "pos", "coref")

# ── Recommended Arabic backbones ─────────────────────────────────────────────
ARABIC_BACKBONES = {
    "arabert":    "aubmindlab/bert-base-arabertv2",
    "camelbert":  "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "araelectra": "aubmindlab/araelectra-base-discriminator",
    "xlmr":       "xlm-roberta-large",
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared BiLSTM Adapter Bridge
# ══════════════════════════════════════════════════════════════════════════════

class SharedBiLSTMBridge(nn.Module):
    """
    Optional shared BiLSTM layer on top of the transformer backbone.

    Sits between the backbone and the NER/POS heads. Improves cross-task
    transfer for sequence labeling by letting the two tasks share a
    recurrent context layer.

    The coref head bypasses this bridge and uses raw transformer outputs
    directly (coref needs fine-grained subword representations).

    Args:
        hidden_size:  Transformer hidden dimension (e.g. 768).
        lstm_hidden:  BiLSTM hidden size per direction (output = 2×).
        dropout:      Dropout on LSTM output.
        num_layers:   Number of BiLSTM layers (1 is sufficient for MTL).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        lstm_hidden: int = 256,
        dropout:     float = 0.1,
        num_layers:  int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.output_dim = 2 * lstm_hidden   # bidirectional

        # Project back to hidden_size so heads don't need to change
        self.projection = nn.Linear(self.output_dim, hidden_size)

    def forward(
        self,
        sequence_output: torch.Tensor,   # (B, S, hidden_size)
        attention_mask:  torch.Tensor,   # (B, S)
    ) -> torch.Tensor:
        """
        Returns:
            (B, S, hidden_size) — same shape as input for drop-in use.
        """
        # Pack padded for efficiency
        lengths = attention_mask.sum(dim=1).cpu()
        packed  = nn.utils.rnn.pack_padded_sequence(
            sequence_output, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=sequence_output.size(1)
        )
        projected = self.projection(self.dropout(unpacked))
        return projected   # (B, S, hidden_size)


# ══════════════════════════════════════════════════════════════════════════════
# Word-level pooling (subword → word alignment)
# ══════════════════════════════════════════════════════════════════════════════

def first_subword_pool(
    sequence_output: torch.Tensor,    # (B, subword_len, H)
    word_ids:        list[list[Optional[int]]],
    max_words:       int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool subword representations to word level using the first-subword
    strategy. Only the first subword of each word contributes.

    Args:
        sequence_output: Subword-level hidden states from backbone.
        word_ids:        List of word_id sequences per example.
                         word_ids[b][j] = word index for subword j,
                         or None for special tokens.
        max_words:       Maximum number of words in the batch.

    Returns:
        word_output:     (B, max_words, H) — word-level representations.
        word_mask:       (B, max_words)    — 1 for real words, 0 for padding.
    """
    B, _, H = sequence_output.shape
    device  = sequence_output.device

    word_output = torch.zeros(B, max_words, H, device=device)
    word_mask   = torch.zeros(B, max_words, dtype=torch.long, device=device)

    for b in range(B):
        seen: set[int] = set()
        for j, wid in enumerate(word_ids[b]):
            if wid is None or wid in seen:
                continue
            if wid < max_words:
                seen.add(wid)
                word_output[b, wid] = sequence_output[b, j]
                word_mask[b, wid]   = 1

    return word_output, word_mask


# ══════════════════════════════════════════════════════════════════════════════
# Unified MTL Model
# ══════════════════════════════════════════════════════════════════════════════

class ArabicMTLModel(nn.Module):
    """
    Unified Multi-Task Learning model for Arabic NER, POS, and Coreference.

    One shared AraBERT backbone + three task-specific heads +
    uncertainty-weighted loss combination (Kendall et al. 2018).

    Args:
        backbone_name:      HuggingFace model name or local path.
        ner_num_labels:     Number of NER tags (default 27 IOB2).
        pos_num_labels:     Number of POS tags (default 32 QCRI).
        hidden_size:        Backbone hidden dimension (auto-detected if None).
        use_bilstm_bridge:  Add shared BiLSTM between backbone and heads.
        lstm_hidden:        BiLSTM hidden size per direction.
        dropout:            Dropout for all heads.
        loss_weighting:     'uncertainty' (Kendall) or 'fixed'.
        fixed_weights:      Dict of fixed weights if loss_weighting='fixed'.
        freeze_backbone_steps: Freeze backbone for first N steps.
        use_morphology:     Enable morphological feature injection in POSHead.
        morph_dim:          Morphological feature dimension.
        max_span_width:     Maximum coreference span width.
        top_lambda:         Fraction of spans kept after mention pruning.
        antecedent_k:       Max antecedent candidates per span.
    """

    def __init__(
        self,
        backbone_name:         str   = "aubmindlab/bert-base-arabertv2",
        ner_num_labels:        int   = 27,
        pos_num_labels:        int   = 32,
        hidden_size:           Optional[int] = None,
        use_bilstm_bridge:     bool  = True,
        lstm_hidden:           int   = 256,
        dropout:               float = 0.1,
        loss_weighting:        str   = "uncertainty",
        fixed_weights:         Optional[dict[str, float]] = None,
        freeze_backbone_steps: int   = 0,
        use_morphology:        bool  = False,
        morph_dim:             int   = 64,
        max_span_width:        int   = 30,
        top_lambda:            float = 0.4,
        antecedent_k:          int   = 50,
    ) -> None:
        super().__init__()

        self.backbone_name          = backbone_name
        self.freeze_backbone_steps  = freeze_backbone_steps
        self._global_step           = 0

        # ── Backbone ──────────────────────────────────────────────────────
        logger.info("Loading backbone: %s", backbone_name)
        config           = AutoConfig.from_pretrained(backbone_name)
        self.backbone    = AutoModel.from_pretrained(backbone_name, config=config)
        self.hidden_size = hidden_size or config.hidden_size

        # ── Shared BiLSTM bridge (optional) ───────────────────────────────
        self.use_bilstm_bridge = use_bilstm_bridge
        if use_bilstm_bridge:
            self.bilstm_bridge = SharedBiLSTMBridge(
                hidden_size=self.hidden_size,
                lstm_hidden=lstm_hidden,
                dropout=dropout,
            )
            logger.info(
                "SharedBiLSTMBridge enabled (hidden=%d, output=%d)",
                lstm_hidden, self.hidden_size,
            )

        # ── Task heads ────────────────────────────────────────────────────
        self.ner_head = NERHead(
            hidden_size=self.hidden_size,
            num_labels=ner_num_labels,
            dropout=dropout,
        )
        self.pos_head = POSHead(
            hidden_size=self.hidden_size,
            num_labels=pos_num_labels,
            dropout=dropout,
            use_morphology=use_morphology,
            morph_dim=morph_dim,
        )
        self.coref_head = CoreferenceHead(
            hidden_size=self.hidden_size,
            max_span_width=max_span_width,
            top_lambda=top_lambda,
            antecedent_k=antecedent_k,
            dropout=dropout,
        )

        # ── Loss weighting ─────────────────────────────────────────────────
        if loss_weighting == "uncertainty":
            self.loss_weighter = UncertaintyWeighter(tasks=list(TASKS))
            logger.info("Using Kendall uncertainty loss weighting")
        else:
            weights = fixed_weights or {"ner": 1.0, "pos": 1.0, "coref": 1.0}
            self.loss_weighter = FixedWeighter(weights=weights)
            logger.info("Using fixed loss weights: %s", weights)

        # ── Freeze backbone if requested ──────────────────────────────────
        if freeze_backbone_steps > 0:
            self._freeze_backbone()
            logger.info(
                "Backbone frozen for first %d steps", freeze_backbone_steps
            )

        # Log parameter counts
        self._log_param_counts()

    # ── Constructor helpers ───────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        backbone_name: str = "aubmindlab/bert-base-arabertv2",
        **kwargs: Any,
    ) -> "ArabicMTLModel":
        """Convenience constructor. All kwargs passed to __init__."""
        return cls(backbone_name=backbone_name, **kwargs)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        backbone_name:   str = "aubmindlab/bert-base-arabertv2",
        **kwargs: Any,
    ) -> "ArabicMTLModel":
        """
        Load a saved model checkpoint.

        Usage:
            model = ArabicMTLModel.load_checkpoint("checkpoints/best_mtl.pt")
        """
        model = cls(backbone_name=backbone_name, **kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model._global_step = state.get("global_step", 0)
        logger.info(
            "Loaded checkpoint from %s (step %d)",
            checkpoint_path, model._global_step,
        )
        return model

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:       torch.Tensor,                    # (B, S)
        attention_mask:  torch.Tensor,                    # (B, S)
        task:            str,                             # "ner"|"pos"|"coref"
        token_type_ids:  Optional[torch.Tensor] = None,   # (B, S)
        word_ids:        Optional[list] = None,           # for word pooling
        ner_labels:      Optional[torch.Tensor] = None,   # (B, W)
        pos_labels:      Optional[torch.Tensor] = None,   # (B, W)
        coref_clusters:  Optional[list] = None,           # list of cluster lists
        morph_features:  Optional[torch.Tensor] = None,   # (B, W, morph_dim)
    ) -> dict[str, Any]:
        """
        Single forward pass through backbone + task head.

        Each batch contains examples from ONE task (task sampling in trainer).
        Only the relevant head runs — no wasted computation.

        Args:
            input_ids:      Tokenized input ids.
            attention_mask: Attention mask (1=real, 0=padding).
            task:           Which task head to activate.
            token_type_ids: Segment ids (BERT-style models only).
            word_ids:       Subword→word mapping from tokenizer.
                            Required for NER and POS (word-level pooling).
                            If None, sequence_output is used directly.
            ner_labels:     NER gold labels (training only).
            pos_labels:     POS gold labels (training only).
            coref_clusters: Gold coreference clusters (training only).
            morph_features: Morphological features for POS head.

        Returns:
            Dict with at minimum:
              "loss"        — scalar tensor (training only)
              "logits"      — raw scores (NER/POS) or mention scores (coref)
              "predictions" — decoded output (tag ids or cluster lists)
              "task"        — echo of the task name
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {TASKS}.")

        # ── Step counter: unfreeze backbone after warmup ───────────────────
        self._global_step += 1
        if (self.freeze_backbone_steps > 0
                and self._global_step == self.freeze_backbone_steps + 1):
            self._unfreeze_backbone()
            logger.info("Backbone unfrozen at step %d", self._global_step)

        # ── 1. Backbone forward pass ───────────────────────────────────────
        backbone_kwargs: dict[str, Any] = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids

        backbone_out    = self.backbone(**backbone_kwargs)
        sequence_output = backbone_out.last_hidden_state   # (B, S, H)

        # ── 2. Word-level pooling for NER and POS ─────────────────────────
        if task in ("ner", "pos"):
            if word_ids is not None:
                max_words = max(
                    (max((wid for wid in wids if wid is not None), default=0) + 1)
                    for wids in word_ids
                )
                word_output, word_mask = first_subword_pool(
                    sequence_output, word_ids, max_words
                )
            else:
                # Fallback: treat subword output as word output
                word_output = sequence_output
                word_mask   = attention_mask

            # ── 3a. Shared BiLSTM bridge (NER + POS only) ─────────────────
            if self.use_bilstm_bridge:
                word_output = self.bilstm_bridge(word_output, word_mask)

        # ── 4. Task head forward ───────────────────────────────────────────
        if task == "ner":
            head_output = self.ner_head(
                word_output, word_mask, labels=ner_labels
            )
        elif task == "pos":
            head_output = self.pos_head(
                word_output, word_mask,
                labels=pos_labels,
                morph_features=morph_features,
            )
        else:  # coref
            head_output = self.coref_head(
                sequence_output, attention_mask, clusters=coref_clusters
            )

        # ── 5. Loss weighting (training only) ─────────────────────────────
        if "loss" in head_output:
            raw_loss    = head_output["loss"]
            weighted    = self.loss_weighter({task: raw_loss})
            head_output["loss"]        = weighted["total_loss"]
            head_output["raw_loss"]    = raw_loss
            head_output["loss_weight"] = weighted.get(f"weight_{task}", 1.0)

        head_output["task"] = task
        return head_output

    # ── Backbone freeze / unfreeze ────────────────────────────────────────

    def _freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("All backbone parameters unfrozen.")

    def freeze_backbone(self) -> None:
        """Public API for manual backbone freezing."""
        self._freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Public API for manual backbone unfreezing."""
        self._unfreeze_backbone()

    # ── Parameter groups for optimizer ────────────────────────────────────

    def get_optimizer_param_groups(
        self,
        backbone_lr:  float = 2e-5,
        head_lr:      float = 4e-5,
        weight_decay: float = 0.01,
    ) -> list[dict[str, Any]]:
        """
        Build optimizer parameter groups with:
          - Backbone:    lower LR (fine-tuning pre-trained weights)
          - Task heads:  higher LR (training from random init)
          - Coref head:  highest LR (most complex head)
          - Bias/LayerNorm: no weight decay (standard practice)

        Usage:
            groups = model.get_optimizer_param_groups(backbone_lr=2e-5)
            optimizer = AdamW(groups)
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

        def has_no_decay(name: str) -> bool:
            return any(nd in name for nd in no_decay)

        # Backbone params
        backbone_params_decay    = []
        backbone_params_no_decay = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if has_no_decay(name):
                backbone_params_no_decay.append(param)
            else:
                backbone_params_decay.append(param)

        # Head params helper
        def head_params(module: nn.Module, lr: float) -> list[dict]:
            decay, no_dec = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (no_dec if has_no_decay(name) else decay).append(param)
            groups = []
            if decay:
                groups.append({
                    "params": decay,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "group": module.__class__.__name__,
                })
            if no_dec:
                groups.append({
                    "params": no_dec,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "group": module.__class__.__name__ + "_no_decay",
                })
            return groups

        groups = [
            {
                "params":       backbone_params_decay,
                "lr":           backbone_lr,
                "weight_decay": weight_decay,
                "group":        "backbone",
            },
            {
                "params":       backbone_params_no_decay,
                "lr":           backbone_lr,
                "weight_decay": 0.0,
                "group":        "backbone_no_decay",
            },
        ]

        # BiLSTM bridge (same LR as backbone — shared representation)
        if self.use_bilstm_bridge:
            groups += head_params(self.bilstm_bridge, backbone_lr)

        # Task heads
        groups += head_params(self.ner_head,   head_lr)
        groups += head_params(self.pos_head,   head_lr)
        groups += head_params(self.coref_head, head_lr * 1.5)  # coref = 1.5×

        # Loss weighter (learnable log_var params)
        groups += head_params(self.loss_weighter, head_lr)

        # Filter empty groups
        return [g for g in groups if g["params"]]

    # ── Checkpoint save / load ────────────────────────────────────────────

    def save_checkpoint(
        self,
        path:        str,
        optimizer:   Optional[Any] = None,
        scheduler:   Optional[Any] = None,
        metrics:     Optional[dict] = None,
    ) -> None:
        """
        Save model + optimizer + scheduler state.

        Usage:
            model.save_checkpoint(
                "checkpoints/best_mtl.pt",
                optimizer=optimizer,
                metrics={"ner_f1": 0.87, "pos_acc": 0.96},
            )
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        state: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
            "global_step":      self._global_step,
            "backbone_name":    self.backbone_name,
            "metrics":          metrics or {},
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            sched_state = (
                scheduler.state_dict()
                if hasattr(scheduler, "state_dict")
                else {}
            )
            state["scheduler_state_dict"] = sched_state

        torch.save(state, path)
        logger.info("Checkpoint saved to %s (step %d)", path, self._global_step)

    # ── Utilities ─────────────────────────────────────────────────────────

    def _log_param_counts(self) -> None:
        total  = sum(p.numel() for p in self.parameters())
        frozen = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        trainable = total - frozen
        logger.info(
            "Model parameters: total=%s  trainable=%s  frozen=%s",
            f"{total:,}", f"{trainable:,}", f"{frozen:,}",
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return total or trainable parameter count."""
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only else self.parameters()
        )
        return sum(p.numel() for p in params)

    def get_backbone_lr_ratio(self) -> float:
        """
        Return the ratio of backbone LR to head LR.
        Used in W&B logging to track discriminative learning rates.
        """
        return 0.5   # backbone_lr = 0.5 × head_lr by default

    def set_task_loss_weights(self, weights: dict[str, float]) -> None:
        """
        Override loss weights at runtime (for ablation studies).

        Usage:
            model.set_task_loss_weights({"ner": 1.0, "pos": 0.5, "coref": 2.0})
        """
        if hasattr(self.loss_weighter, "weights"):
            self.loss_weighter.weights.update(weights)
            logger.info("Loss weights updated: %s", weights)
        else:
            logger.warning(
                "Cannot set weights on UncertaintyWeighter directly. "
                "Switch to loss_weighting='fixed' for manual weights."
            )

    def summary(self) -> str:
        """Return a human-readable model summary."""
        lines = [
            "─" * 60,
            "  ArabicMTLModel",
            f"  Backbone     : {self.backbone_name}",
            f"  Hidden size  : {self.hidden_size}",
            f"  BiLSTM bridge: {self.use_bilstm_bridge}",
            f"  NER head     : {self.ner_head.num_labels} labels",
            f"  POS head     : {self.pos_head.num_labels} labels",
            f"  Coref head   : max_span={self.coref_head.max_span_width}",
            f"  Loss weighter: {self.loss_weighter.__class__.__name__}",
            f"  Total params : {self.num_parameters():,}",
            f"  Trainable    : {self.num_parameters(trainable_only=True):,}",
            "─" * 60,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print("\n── ArabicMTLModel architecture check ───────────────────")
    print("(uses a tiny BERT config — no HuggingFace download needed)\n")

    from transformers import BertConfig, BertModel

    # ── Build a tiny mock model without downloading AraBERT ──────────────
    class _MockMTL(ArabicMTLModel):
        """Tiny version for testing — swaps AraBERT for a 2-layer BERT."""
        def __init__(self):
            # Skip parent __init__ to avoid HF download
            nn.Module.__init__(self)
            self.backbone_name         = "mock-bert"
            self.hidden_size           = 128
            self.use_bilstm_bridge     = True
            self.freeze_backbone_steps = 0
            self._global_step          = 0

            cfg = BertConfig(
                hidden_size=128, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=256,
            )
            self.backbone      = BertModel(cfg)
            self.bilstm_bridge = SharedBiLSTMBridge(128, 64, dropout=0.0)

            self.ner_head   = NERHead(hidden_size=128, num_labels=27)
            self.pos_head   = POSHead(hidden_size=128, num_labels=32)
            self.coref_head = CoreferenceHead(
                hidden_size=128, max_span_width=5, antecedent_k=5
            )
            self.loss_weighter = UncertaintyWeighter(tasks=list(TASKS))

    model = _MockMTL()
    model.eval()

    B, S = 2, 20
    ids   = torch.randint(0, 1000, (B, S))
    mask  = torch.ones(B, S, dtype=torch.long)
    mask[1, 15:] = 0

    # word_ids: simulate 15 words for batch[0], 12 for batch[1]
    word_ids = [
        [None] + [i // 1 for i in range(15)] + [None] * (S - 16),
        [None] + [i // 1 for i in range(12)] + [None] * (S - 13),
    ]
    word_ids[0] = word_ids[0][:S]
    word_ids[1] = word_ids[1][:S]

    errors = []
    with torch.no_grad():
        # NER
        ner_labels = torch.randint(0, 27, (B, 15))
        out = model(ids, mask, task="ner",
                    word_ids=word_ids, ner_labels=ner_labels)
        assert "loss" in out and "predictions" in out, "NER output missing keys"
        print(f"  ✅ NER  — loss={out['loss'].item():.4f}  "
              f"preds={out['predictions'].shape}")

        # POS
        pos_labels = torch.randint(0, 32, (B, 15))
        out = model(ids, mask, task="pos",
                    word_ids=word_ids, pos_labels=pos_labels)
        assert "loss" in out and "predictions" in out, "POS output missing keys"
        print(f"  ✅ POS  — loss={out['loss'].item():.4f}  "
              f"preds={out['predictions'].shape}")

        # Coref
        clusters = [[[[0, 1], [5, 6]]], [[[0, 0], [3, 3]]]]
        out = model(ids, mask, task="coref", coref_clusters=clusters)
        assert "loss" in out and "predictions" in out, "Coref output missing keys"
        print(f"  ✅ Coref — loss={out['loss'].item():.4f}  "
              f"clusters={[len(c) for c in out['predictions']]}")

    # Optimizer param groups
    groups = model.get_optimizer_param_groups()
    total_params_in_groups = sum(
        sum(p.numel() for p in g["params"]) for g in groups
    )
    print(f"\n  ✅ Optimizer groups: {len(groups)}")
    print(f"     Params covered : {total_params_in_groups:,}")

    print(f"\n{model.summary()}")
    print("\n  mtl_model.py ready ✅\n")