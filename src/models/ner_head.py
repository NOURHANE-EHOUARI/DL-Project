"""
ner_head.py — BiLSTM-CRF Named Entity Recognition Head.

Owner:   Student A
Phase:   3 — Modeling & Training (Week 3–5)

Architecture:
  backbone hidden states
       ↓
  dropout
       ↓
  BiLSTM  (optional — adds contextual smoothing on top of transformer)
       ↓
  linear projection → emission scores  (batch, seq_len, num_tags)
       ↓
  CRF layer  (Viterbi decode at inference / neg-log-likelihood loss at train)

Tag scheme: IOB2  (B-TYPE, I-TYPE, O)
Tagset built from ANERcorp / Wojood label set — see LABELS below.

References:
  - Lafferty et al. (2001) CRF
  - Huang et al. (2015) BiLSTM-CRF
  - Ma & Hovy (2016) end-to-end NER
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchcrf import CRF  # pip install pytorch-crf

# ---------------------------------------------------------------------------
# IOB2 label set  (ANERcorp + Wojood compatible)
# ---------------------------------------------------------------------------
LABELS: list[str] = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-GPE", "I-GPE",
    "B-DATE", "I-DATE",
    "B-TIME", "I-TIME",
    "B-MONEY", "I-MONEY",
    "B-PERCENT", "I-PERCENT",
    "B-PRODUCT", "I-PRODUCT",
    "B-EVENT", "I-EVENT",
    "B-NORP", "I-NORP",
    "B-FAC", "I-FAC",
    "B-LANGUAGE", "I-LANGUAGE",
]

LABEL2ID: dict[str, int] = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(LABELS)}
NUM_LABELS: int = len(LABELS)

# ID used by DataCollator / CrossEntropyLoss to ignore padding positions
IGNORE_INDEX: int = -100


# ---------------------------------------------------------------------------
# NER Head
# ---------------------------------------------------------------------------
class NERHead(nn.Module):
    """
    BiLSTM-CRF head for Arabic Named Entity Recognition.

    Args:
        hidden_size:   Dimension of backbone output (e.g. 768 for AraBERT).
        num_labels:    Number of IOB2 tags (default = len(LABELS)).
        lstm_hidden:   Hidden size of the BiLSTM layer (0 = skip BiLSTM).
        lstm_layers:   Number of BiLSTM stacking layers.
        dropout:       Dropout applied after backbone / BiLSTM.
        use_morphology: If True, expects extra `morph_features` tensor
                        concatenated before the projection layer.
        morph_dim:     Dimension of morphological feature vector per token.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = NUM_LABELS,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        use_morphology: bool = False,
        morph_dim: int = 64,
    ) -> None:
        super().__init__()

        self.use_bilstm = lstm_hidden > 0
        self.use_morphology = use_morphology
        self.num_labels = num_labels

        self.dropout = nn.Dropout(dropout)

        # ── Optional BiLSTM ──────────────────────────────────────────────
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            proj_in = lstm_hidden * 2  # bidirectional
        else:
            self.bilstm = None
            proj_in = hidden_size

        # ── Optional morphological feature injection ─────────────────────
        if use_morphology:
            self.morph_proj = nn.Linear(morph_dim, morph_dim)
            proj_in += morph_dim

        # ── Emission score projection ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_in, proj_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_in // 2, num_labels),
        )

        # ── CRF layer ────────────────────────────────────────────────────
        self.crf = CRF(num_labels, batch_first=True)

        self._init_weights()

    # ── Weight init ──────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "crf" in name:
                continue  # CRF has its own init
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ── Core forward ─────────────────────────────────────────────────────
    def _encode(
        self,
        sequence_output: torch.Tensor,       # (B, S, H) — word-level from backbone
        morph_features: torch.Tensor | None, # (B, S, morph_dim) — optional
    ) -> torch.Tensor:
        """Compute emission scores from backbone output."""
        x = self.dropout(sequence_output)

        if self.use_bilstm:
            x, _ = self.bilstm(x)             # (B, S, 2*lstm_hidden)
            x = self.dropout(x)

        if self.use_morphology and morph_features is not None:
            m = self.morph_proj(morph_features)
            m = torch.relu(m)
            x = torch.cat([x, m], dim=-1)    # (B, S, proj_in)

        emissions = self.classifier(x)        # (B, S, num_labels)
        return emissions

    # ── Training loss ─────────────────────────────────────────────────────
    def forward(
        self,
        sequence_output: torch.Tensor,        # (B, S, H) — word-aligned hidden states
        attention_mask: torch.Tensor,         # (B, S)    — word-level mask (1=real, 0=pad)
        labels: torch.Tensor | None = None,   # (B, S)    — IOB2 label ids (-100 = ignore)
        morph_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sequence_output:  Word-level hidden states from backbone.get_word_outputs().
            attention_mask:   Word-level boolean mask (True = valid token).
            labels:           IOB2 label ids for loss computation.
                              Use IGNORE_INDEX (-100) for sub-word / padding positions.
            morph_features:   Optional Farasa morphological feature vectors.

        Returns:
            dict with keys:
              "loss"       — CRF neg-log-likelihood (only if labels provided)
              "logits"     — emission scores  (B, S, num_labels)
              "predictions"— Viterbi-decoded tag sequences as list[list[int]]
        """
        bool_mask = attention_mask.bool()     # CRF expects BoolTensor
        emissions = self._encode(sequence_output, morph_features)

        output: dict[str, torch.Tensor] = {"logits": emissions}

        if labels is not None:
            # Replace IGNORE_INDEX with "O" (0) so CRF doesn't crash on padding
            # but mask those positions out via bool_mask.
            crf_labels = labels.clone()
            crf_labels[crf_labels == IGNORE_INDEX] = 0

            # CRF returns log-likelihood (positive) — negate for loss
            log_likelihood = self.crf(emissions, crf_labels, mask=bool_mask, reduction="mean")
            output["loss"] = -log_likelihood

        # Viterbi decoding — always run (needed for eval + MTL model predictions)
        predictions = self.crf.decode(emissions, mask=bool_mask)
        output["predictions"] = predictions   # list[list[int]], variable length

        return output

    # ── Convenience: entity-level prediction ─────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor,
        morph_features: torch.Tensor | None = None,
    ) -> list[list[str]]:
        """
        Returns predicted tag strings (not ids).
        Useful for inference and error analysis.

        Example:
            tags = ner_head.predict(word_output, word_mask)
            # [["B-PER", "I-PER", "O", "B-LOC", ...], ...]
        """
        self.eval()
        out = self.forward(sequence_output, attention_mask,
                           morph_features=morph_features)
        return [
            [ID2LABEL[tag_id] for tag_id in seq]
            for seq in out["predictions"]
        ]


# ---------------------------------------------------------------------------
# IOB2 utilities
# ---------------------------------------------------------------------------

def iob2_to_spans(
    tags: list[str], tokens: list[str] | None = None
) -> list[dict]:
    """
    Convert a flat IOB2 tag sequence to a list of entity span dicts.

    Args:
        tags:   e.g. ["O", "B-PER", "I-PER", "O", "B-LOC"]
        tokens: optional list of word strings for human-readable output

    Returns:
        list of {"type": str, "start": int, "end": int, "text": str | None}
        where start/end are inclusive word indices.

    Example:
        tags   = ["O", "B-PER", "I-PER", "O"]
        tokens = ["في", "محمد", "صلاح", "اليوم"]
        → [{"type": "PER", "start": 1, "end": 2, "text": "محمد صلاح"}]
    """
    spans = []
    current_type = None
    current_start = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            # Close previous span if open
            if current_type is not None:
                spans.append(_make_span(current_type, current_start, i - 1, tokens))
            current_type = tag[2:]
            current_start = i
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_type != entity_type:
                # Malformed I- without matching B- — treat as new entity
                if current_type is not None:
                    spans.append(_make_span(current_type, current_start, i - 1, tokens))
                current_type = entity_type
                current_start = i
            # else: continuation — do nothing
        else:  # "O"
            if current_type is not None:
                spans.append(_make_span(current_type, current_start, i - 1, tokens))
                current_type = None
                current_start = None

    # Close final span
    if current_type is not None:
        spans.append(_make_span(current_type, current_start, len(tags) - 1, tokens))

    return spans


def _make_span(
    entity_type: str, start: int, end: int, tokens: list[str] | None
) -> dict:
    text = " ".join(tokens[start: end + 1]) if tokens is not None else None
    return {"type": entity_type, "start": start, "end": end, "text": text}


def validate_iob2(tags: list[str]) -> list[str]:
    """
    Fix invalid IOB2 sequences in-place (I- without B-).
    Returns corrected tag list.
    """
    corrected = []
    prev_type = None
    for tag in tags:
        if tag.startswith("I-"):
            entity_type = tag[2:]
            if prev_type != entity_type:
                tag = "B-" + entity_type   # fix: convert to B-
        corrected.append(tag)
        prev_type = tag[2:] if (tag.startswith("B-") or tag.startswith("I-")) else None
    return corrected


# ---------------------------------------------------------------------------
# Quick sanity check  (run: python ner_head.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("=== NER Head Sanity Check ===\n")

    # ── 1. Label sanity ──────────────────────────────────────────────────
    assert all(lbl.startswith("B-") or lbl.startswith("I-") or lbl == "O"
               for lbl in LABELS), "Bad label found"
    print(f"✓ {NUM_LABELS} IOB2 labels registered")

    # ── 2. IOB2 span extraction ──────────────────────────────────────────
    tags = ["O", "B-PER", "I-PER", "O", "B-LOC", "O"]
    toks = ["في", "محمد", "صلاح", "زار", "القاهرة", "اليوم"]
    spans = iob2_to_spans(tags, toks)
    assert len(spans) == 2
    assert spans[0]["type"] == "PER" and spans[0]["text"] == "محمد صلاح"
    assert spans[1]["type"] == "LOC" and spans[1]["text"] == "القاهرة"
    print(f"✓ iob2_to_spans: {spans}")

    # ── 3. validate_iob2 ────────────────────────────────────────────────
    broken = ["O", "I-PER", "I-PER", "O"]
    fixed = validate_iob2(broken)
    assert fixed == ["O", "B-PER", "I-PER", "O"]
    print(f"✓ validate_iob2: {broken} → {fixed}")

    # ── 4. Model forward pass ────────────────────────────────────────────
    try:
        B, S, H = 2, 10, 768
        head = NERHead(hidden_size=H, lstm_hidden=256, dropout=0.1)
        head.eval()

        seq_out = torch.randn(B, S, H)
        mask = torch.ones(B, S, dtype=torch.long)
        mask[1, 7:] = 0   # batch item 1 is shorter

        labels = torch.randint(0, NUM_LABELS, (B, S))
        labels[1, 7:] = IGNORE_INDEX

        with torch.no_grad():
            out_train = head(seq_out, mask, labels=labels)
            out_infer = head(seq_out, mask)

        assert "loss" in out_train
        assert "logits" in out_train
        assert len(out_infer["predictions"]) == B

        print(f"✓ forward (train): loss={out_train['loss'].item():.4f}")
        print(f"✓ forward (infer): {len(out_infer['predictions'])} sequences decoded")

        # predict() helper
        tag_strings = head.predict(seq_out, mask)
        assert all(isinstance(t, str) for seq in tag_strings for t in seq)
        print(f"✓ predict(): sample = {tag_strings[0][:5]}")

    except ImportError:
        print("⚠  pytorch-crf not installed — run: pip install pytorch-crf")
        print("   Model structure OK, skipping forward pass test.")

    print("\n✅ All checks passed — place in src/models/ner_head.py")