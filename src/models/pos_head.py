"""
pos_head.py — Token Classification Head for Arabic POS Tagging.
Owner:   Student B (Hiba)
Phase:   3 — Modeling & Training (Week 3–5)

Architecture:
  backbone word-level hidden states
       ↓
  dropout
       ↓
  optional morphological feature injection (concatenation)
       ↓
  linear projection → logits  (batch, seq_len, num_tags)
       ↓
  CrossEntropyLoss (training) / argmax (inference)

Tagset: QCRI Arabic Dialect POS tagset
  Covers MSA + dialectal tags: NOUN, VERB, ADJ, PREP, PRON,
  CONJ, PART, ADV, NUM, PUNC + compound tags (NOUN+PRON, etc.)

References:
  - Abandah et al. (2015) Arabic POS tagging
  - Darwish et al. (2017) Arabic NLP
  - Inoue et al. (2022) CAMeLBERT for Arabic NLP tasks
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── POS Tagset ─────────────────────────────────────────────────────────────────
# Core QCRI dialect tagset + compound tags from morphological segmentation
# Covers Egyptian, Gulf, Levantine, Maghrebi dialects + MSA
LABELS: list[str] = [
    # Core POS tags
    "NOUN", "V", "ADJ", "PREP", "PRON", "CONJ", "PART",
    "ADV", "NUM", "PUNC", "DET", "ABBREV", "FOREIGN",
    # Compound tags (clitic + base, common in Arabic)
    "NOUN+PRON", "V+PRON", "PREP+PRON", "CONJ+PART",
    "DET+NOUN", "PREP+DET", "CONJ+NOUN", "CONJ+V",
    "FUT_PART+V", "PROG_PART+V", "PROG_PART+V+PRON",
    "FUT_PART+V+PRON", "CONJ+ADJ", "ADJ+PRON",
    "CONJ+PRON", "PART+PRON", "NOUN+PREP+PRON",
    # Fallback for unseen tags
    "OTHER",
]

LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABELS)}
NUM_LABELS: int = len(LABELS)
IGNORE_INDEX: int = -100


def get_label2id(extra_tags: list[str] | None = None) -> dict[str, int]:
    """
    Build a label2id mapping, optionally extending with extra tags.
    Use this when loading a dataset with tags not in the default set.

    Args:
        extra_tags: additional tags to add beyond the default set

    Returns:
        Extended label2id dict
    """
    mapping = dict(LABEL2ID)
    if extra_tags:
        for tag in extra_tags:
            if tag not in mapping:
                mapping[tag] = len(mapping)
    return mapping


# ══════════════════════════════════════════════════════════════════════════════
# POS Head
# ══════════════════════════════════════════════════════════════════════════════

class POSHead(nn.Module):
    """
    Token classification head for Arabic Part-of-Speech tagging.

    Designed to plug directly onto ArabicBackbone.get_word_outputs().
    Follows the same interface as NERHead for clean MTL integration.

    Args:
        hidden_size:     Backbone output dimension (e.g. 768 for AraBERT).
        num_labels:      Number of POS tags (default = len(LABELS)).
        intermediate_size: Hidden size of intermediate projection.
                           Set to 0 to use a single linear layer.
        dropout:         Dropout probability.
        use_morphology:  If True, expects morph_features tensor
                         (innovative feature #2 from project blueprint).
        morph_dim:       Dimension of morphological feature vectors.
        label2id:        Custom label2id mapping (overrides default LABEL2ID).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = NUM_LABELS,
        intermediate_size: int = 256,
        dropout: float = 0.1,
        use_morphology: bool = False,
        morph_dim: int = 64,
        label2id: dict[str, int] | None = None,
    ) -> None:
        super().__init__()

        self.num_labels      = num_labels
        self.use_morphology  = use_morphology
        self.label2id        = label2id or LABEL2ID
        self.id2label        = {v: k for k, v in self.label2id.items()}

        self.dropout = nn.Dropout(dropout)

        # ── Morphological feature projection ──────────────────────────────
        if use_morphology:
            self.morph_proj = nn.Sequential(
                nn.Linear(morph_dim, morph_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            classifier_in = hidden_size + morph_dim
        else:
            self.morph_proj  = None
            classifier_in    = hidden_size

        # ── Token classifier ──────────────────────────────────────────────
        if intermediate_size > 0:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, intermediate_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, num_labels),
            )
        else:
            # Lightweight: single linear (faster, still strong on top of BERT)
            self.classifier = nn.Linear(classifier_in, num_labels)

        # ── Loss function ─────────────────────────────────────────────────
        # ignore_index=-100 matches the collator padding value
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        self._init_weights()

    # ── Weight initialization ─────────────────────────────────────────────
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Core forward ──────────────────────────────────────────────────────
    def forward(
        self,
        sequence_output: torch.Tensor,              # (B, S, H)
        attention_mask:  torch.Tensor,              # (B, S)
        labels:          Optional[torch.Tensor] = None,  # (B, S)
        morph_features:  Optional[torch.Tensor] = None,  # (B, S, morph_dim)
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sequence_output: Word-level hidden states from
                             ArabicBackbone.get_word_outputs().
                             Shape: (batch, max_words, hidden_size)
            attention_mask:  Word-level mask (1=real word, 0=padding).
                             Shape: (batch, max_words)
            labels:          POS tag ids. IGNORE_INDEX (-100) for padding.
                             Shape: (batch, max_words)
            morph_features:  Optional morphological feature vectors per token.
                             Shape: (batch, max_words, morph_dim)

        Returns:
            dict with keys:
              "loss"        — CrossEntropy loss (only if labels provided)
              "logits"      — raw scores (batch, max_words, num_labels)
              "predictions" — argmax tag ids (batch, max_words)
        """
        x = self.dropout(sequence_output)           # (B, S, H)

        # Inject morphological features if available
        if self.use_morphology and morph_features is not None:
            m = self.morph_proj(morph_features)     # (B, S, morph_dim)
            x = torch.cat([x, m], dim=-1)           # (B, S, H + morph_dim)

        logits = self.classifier(x)                 # (B, S, num_labels)

        output: dict[str, torch.Tensor] = {"logits": logits}

        # Loss — only computed if labels are provided (training / dev)
        if labels is not None:
            # CrossEntropyLoss expects (B*S, num_labels) and (B*S,)
            loss = self.loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )
            output["loss"] = loss

        # Predictions — mask out padding positions
        predictions = logits.argmax(dim=-1)         # (B, S)
        # Zero-out padding positions so they don't mislead evaluation
        predictions = predictions * attention_mask
        output["predictions"] = predictions

        return output

    # ── Convenience: string predictions ──────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        sequence_output:  torch.Tensor,
        attention_mask:   torch.Tensor,
        morph_features:   Optional[torch.Tensor] = None,
    ) -> list[list[str]]:
        """
        Returns predicted POS tag strings per sentence.
        Padding positions (attention_mask=0) are excluded.

        Example:
            tags = pos_head.predict(word_output, word_mask)
            # [["NOUN", "V", "PREP", "NOUN"], ["ADJ", "NOUN", ...]]
        """
        self.eval()
        out = self.forward(sequence_output, attention_mask,
                           morph_features=morph_features)
        predictions = out["predictions"]            # (B, S)

        tag_sequences = []
        for b in range(predictions.size(0)):
            seq_len = attention_mask[b].sum().item()
            tag_ids = predictions[b, :int(seq_len)].tolist()
            tags    = [self.id2label.get(tid, "OTHER") for tid in tag_ids]
            tag_sequences.append(tags)

        return tag_sequences

    # ── Accuracy metrics ──────────────────────────────────────────────────
    @torch.no_grad()
    def compute_accuracy(
        self,
        logits:  torch.Tensor,   # (B, S, num_labels)
        labels:  torch.Tensor,   # (B, S)
        mask:    torch.Tensor,   # (B, S)
    ) -> dict[str, float]:
        """
        Compute token-level accuracy on real (non-padding) positions.
        Used during training for quick per-batch monitoring.

        Returns:
            {"accuracy": float, "correct": int, "total": int}
        """
        predictions = logits.argmax(dim=-1)          # (B, S)
        # Only consider non-padding, non-ignored positions
        valid = (labels != IGNORE_INDEX) & mask.bool()
        correct = ((predictions == labels) & valid).sum().item()
        total   = valid.sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": int(correct), "total": int(total)}


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== POS Head Sanity Check ===\n")

    B, S, H = 2, 15, 768

    # ── Basic forward pass ────────────────────────────────────────────────
    head = POSHead(hidden_size=H, intermediate_size=256, dropout=0.1)
    head.eval()

    seq_out = torch.randn(B, S, H)
    mask    = torch.ones(B, S, dtype=torch.long)
    mask[1, 10:] = 0   # second sentence is shorter

    labels  = torch.randint(0, NUM_LABELS, (B, S))
    labels[1, 10:] = IGNORE_INDEX

    with torch.no_grad():
        out_train = head(seq_out, mask, labels=labels)
        out_infer = head(seq_out, mask)

    assert "loss"        in out_train, "loss missing"
    assert "logits"      in out_train, "logits missing"
    assert "predictions" in out_train, "predictions missing"
    assert out_train["logits"].shape == (B, S, NUM_LABELS)
    print(f"✓ forward (train): loss={out_train['loss'].item():.4f}")
    print(f"✓ forward (infer): logits={out_infer['logits'].shape}")

    # ── Morphological feature injection ──────────────────────────────────
    morph_head = POSHead(hidden_size=H, use_morphology=True, morph_dim=64)
    morph_feats = torch.randn(B, S, 64)
    with torch.no_grad():
        out_morph = morph_head(seq_out, mask, labels=labels,
                               morph_features=morph_feats)
    assert "loss" in out_morph
    print(f"✓ morphology injection: loss={out_morph['loss'].item():.4f}")

    # ── predict() helper ──────────────────────────────────────────────────
    tag_strings = head.predict(seq_out, mask)
    assert len(tag_strings) == B
    assert all(isinstance(t, str) for seq in tag_strings for t in seq)
    print(f"✓ predict(): {tag_strings[0][:5]}")

    # ── Accuracy ──────────────────────────────────────────────────────────
    acc = head.compute_accuracy(out_train["logits"], labels, mask)
    print(f"✓ accuracy: {acc['accuracy']:.2%} "
          f"({acc['correct']}/{acc['total']} tokens)")

    # ── Label sanity ──────────────────────────────────────────────────────
    assert len(LABEL2ID) == len(ID2LABEL) == NUM_LABELS
    assert LABEL2ID["NOUN"] == 0
    print(f"✓ {NUM_LABELS} POS tags registered")
    print(f"✓ Sample tags: {LABELS[:8]}")

    # ── Custom label2id ───────────────────────────────────────────────────
    custom = get_label2id(extra_tags=["DIALECT_SPECIFIC", "UNKNOWN"])
    assert "DIALECT_SPECIFIC" in custom
    assert len(custom) == NUM_LABELS + 2
    print(f"✓ get_label2id() with extra tags: {len(custom)} total tags")

    print("\n✅ All checks passed — src/models/pos_head.py ready")
