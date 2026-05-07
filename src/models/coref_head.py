"""
coref_head.py — End-to-End Coreference Resolution Head for Arabic.
Owner:   Student B (Hiba)
Phase:   3 — Modeling & Training (Week 3–5)

Architecture (Lee et al. 2018 — adapted for Arabic):
  backbone hidden states (subword level)
       ↓
  span representation  =  [h_start ; h_end ; h_attended ; span_width_emb]
       ↓
  mention scorer  →  prune spans to top-k candidates
       ↓
  antecedent scorer  →  for each span, score all previous spans
       ↓
  coreference clusters via highest-scoring antecedent links

Key adaptations for Arabic:
  - Span width embedding captures Arabic morphological span patterns
  - Compatible with AraBERT subword tokenization
  - Integrates with ArabicBackbone.forward() output directly

References:
  - Lee et al. (2018) "End-to-end Neural Coreference Resolution" EMNLP
  - Kantor & Globerson (2019) span representation improvements
  - Inoue et al. (2020) "Coreference Resolution for Arabic" ACL
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

IGNORE_INDEX: int = -100


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def batch_select(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Select rows from tensor using indices per batch item.

    Args:
        tensor:  (batch, seq, hidden)
        indices: (batch, k) — indices to gather

    Returns:
        (batch, k, hidden)
    """
    batch, k = indices.shape
    hidden   = tensor.shape[-1]
    expanded = indices.unsqueeze(-1).expand(batch, k, hidden)
    return torch.gather(tensor, 1, expanded)


def get_span_mask(
    span_starts: torch.Tensor,   # (batch, num_spans)
    span_ends:   torch.Tensor,   # (batch, num_spans)
    seq_len:     int,
) -> torch.Tensor:
    """Return a boolean mask for valid spans (start <= end < seq_len)."""
    valid_start = span_starts >= 0
    valid_end   = span_ends < seq_len
    valid_order = span_starts <= span_ends
    return valid_start & valid_end & valid_order


# ══════════════════════════════════════════════════════════════════════════════
# Span Representation
# ══════════════════════════════════════════════════════════════════════════════

class SpanRepresentation(nn.Module):
    """
    Builds a fixed-size vector for each candidate mention span.

    Span vector = [h_start ; h_end ; h_attended ; phi(width)]

    where:
      h_start    = hidden state of first token in span
      h_end      = hidden state of last token in span
      h_attended = attention-weighted sum of tokens in span
      phi(width) = learned embedding of span width (number of tokens)

    This gives 3*H + span_width_emb_dim dimensions total.

    Args:
        hidden_size:        Backbone hidden size H.
        span_width_emb_dim: Embedding dimension for span widths.
        max_span_width:     Maximum allowed span width in tokens.
        dropout:            Dropout on span representations.
    """

    def __init__(
        self,
        hidden_size:        int = 768,
        span_width_emb_dim: int = 64,
        max_span_width:     int = 30,
        dropout:            float = 0.3,
    ) -> None:
        super().__init__()

        self.hidden_size        = hidden_size
        self.max_span_width     = max_span_width
        self.span_width_emb_dim = span_width_emb_dim

        # Attention over tokens within a span (scalar attention)
        self.span_attn = nn.Linear(hidden_size, 1)

        # Width embedding: one vector per possible span width
        self.width_embedding = nn.Embedding(max_span_width + 1, span_width_emb_dim)

        self.dropout = nn.Dropout(dropout)

        # Output dimension
        self.output_dim = 3 * hidden_size + span_width_emb_dim

    def forward(
        self,
        sequence_output: torch.Tensor,   # (batch, seq_len, H)
        span_starts:     torch.Tensor,   # (batch, num_spans)
        span_ends:       torch.Tensor,   # (batch, num_spans)
    ) -> torch.Tensor:
        """
        Args:
            sequence_output: Subword-level hidden states from backbone.
            span_starts:     Start token indices of candidate spans.
            span_ends:       End token indices of candidate spans (inclusive).

        Returns:
            span_repr: (batch, num_spans, output_dim)
        """
        batch, seq_len, H = sequence_output.shape
        num_spans = span_starts.shape[1]
        device    = sequence_output.device

        # ── Start and end token representations ───────────────────────────
        # Clamp indices to valid range
        starts_clamped = span_starts.clamp(0, seq_len - 1)  # (B, num_spans)
        ends_clamped   = span_ends.clamp(0, seq_len - 1)

        h_start = batch_select(sequence_output, starts_clamped)  # (B, num_spans, H)
        h_end   = batch_select(sequence_output, ends_clamped)    # (B, num_spans, H)

        # ── Attended span representation ──────────────────────────────────
        # For each span, compute attention over its tokens and take
        # a weighted sum. This is O(batch * num_spans * seq_len) —
        # acceptable for small num_spans after pruning.

        # Attention scores for all token positions: (B, seq_len, 1)
        attn_scores = self.span_attn(self.dropout(sequence_output))  # (B, S, 1)
        attn_scores = attn_scores.squeeze(-1)                         # (B, S)

        # Build span masks: for span i, mask[i, j] = 1 if start_i <= j <= end_i
        # Shape: (B, num_spans, seq_len)
        token_idx = torch.arange(seq_len, device=device)             # (S,)
        token_idx = token_idx.unsqueeze(0).unsqueeze(0)              # (1, 1, S)
        s_exp     = span_starts.unsqueeze(-1)                         # (B, num_spans, 1)
        e_exp     = span_ends.unsqueeze(-1)                           # (B, num_spans, 1)
        span_mask = (token_idx >= s_exp) & (token_idx <= e_exp)      # (B, num_spans, S)

        # Masked softmax
        attn_exp = attn_scores.unsqueeze(1).expand_as(span_mask)     # (B, num_spans, S)
        attn_exp = attn_exp.masked_fill(~span_mask, float("-inf"))
        attn_weights = F.softmax(attn_exp, dim=-1)                   # (B, num_spans, S)
        # Handle all-masked spans (all -inf → NaN after softmax)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Weighted sum: (B, num_spans, H)
        h_attended = torch.bmm(attn_weights, sequence_output)

        # ── Span width embedding ──────────────────────────────────────────
        widths = (span_ends - span_starts).clamp(0, self.max_span_width)
        width_emb = self.width_embedding(widths)                     # (B, num_spans, W)

        # ── Concatenate all components ────────────────────────────────────
        span_repr = torch.cat([h_start, h_end, h_attended, width_emb], dim=-1)
        return self.dropout(span_repr)                               # (B, num_spans, 3H+W)


# ══════════════════════════════════════════════════════════════════════════════
# Mention Scorer
# ══════════════════════════════════════════════════════════════════════════════

class MentionScorer(nn.Module):
    """
    Scores candidate spans as potential mentions.
    Used to prune the O(T²) candidate set to top-λT spans.

    Args:
        input_dim:  Span representation dimension (3H + W).
        hidden_dim: Hidden layer size.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 150,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, span_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            span_repr: (batch, num_spans, input_dim)

        Returns:
            scores: (batch, num_spans) — scalar score per span
        """
        return self.scorer(span_repr).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Antecedent Scorer
# ══════════════════════════════════════════════════════════════════════════════

class AntecedentScorer(nn.Module):
    """
    Scores (mention, antecedent) pairs for coreference.

    Score(i, j) = mention_score(i) + mention_score(j) + pair_score(i, j)

    where pair_score is a bilinear/feedforward scorer over the
    concatenated span representations.

    Args:
        input_dim:  Span representation dimension.
        hidden_dim: Hidden layer for pairwise scorer.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 150,
        dropout:    float = 0.3,
    ) -> None:
        super().__init__()

        # Pairwise scorer: takes [span_i ; span_j ; span_i * span_j]
        self.pair_scorer = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        span_repr:       torch.Tensor,   # (batch, num_spans, input_dim)
        mention_scores:  torch.Tensor,   # (batch, num_spans)
        top_k:           int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each span, score its top-k antecedent candidates.

        Args:
            span_repr:      Span representations.
            mention_scores: Scalar mention scores per span.
            top_k:          Number of antecedent candidates to consider.

        Returns:
            antecedent_scores: (batch, num_spans, top_k + 1)
                               +1 for the "no antecedent" (epsilon) option.
            top_k_indices:     (batch, num_spans, top_k) — antecedent indices.
            antecedent_mask:   (batch, num_spans, top_k) — valid antecedent mask.
        """
        batch, num_spans, input_dim = span_repr.shape
        device = span_repr.device

        # ── Select top-k antecedent candidates (previous spans) ───────────
        # For span i, antecedents are spans j < i (enforce ordering)
        k = min(top_k, num_spans - 1)

        # Build antecedent index matrix: for each span i, candidates are
        # the k spans immediately before it
        span_idx  = torch.arange(num_spans, device=device)          # (S,)
        top_k_idx = span_idx.unsqueeze(1) - torch.arange(
            1, k + 1, device=device
        ).unsqueeze(0)                                               # (S, k)
        top_k_idx = top_k_idx.clamp(min=0)                          # (S, k)
        top_k_idx = top_k_idx.unsqueeze(0).expand(batch, -1, -1)   # (B, S, k)

        # Valid antecedent mask: antecedent index must be < span index
        valid_mask = (
            top_k_idx < span_idx.unsqueeze(1).unsqueeze(0)
        )                                                            # (B, S, k)

        # ── Gather antecedent representations ────────────────────────────
        flat_idx = top_k_idx.reshape(batch, -1)                     # (B, S*k)
        ant_repr = batch_select(span_repr, flat_idx)                # (B, S*k, H)
        ant_repr = ant_repr.reshape(batch, num_spans, k, input_dim) # (B, S, k, H)

        # ── Pairwise scoring ──────────────────────────────────────────────
        span_exp = span_repr.unsqueeze(2).expand_as(ant_repr)       # (B, S, k, H)
        pair_inp = torch.cat(
            [span_exp, ant_repr, span_exp * ant_repr], dim=-1
        )                                                            # (B, S, k, 3H)
        pair_scores = self.pair_scorer(pair_inp).squeeze(-1)        # (B, S, k)

        # ── Full antecedent score = mention(i) + mention(j) + pair(i,j) ──
        ant_mention_scores = torch.gather(
            mention_scores.unsqueeze(1).expand(batch, num_spans, num_spans),
            2,
            top_k_idx,
        )                                                            # (B, S, k)

        span_mention_exp = mention_scores.unsqueeze(2).expand_as(pair_scores)
        total_scores = span_mention_exp + ant_mention_scores + pair_scores
        total_scores = total_scores.masked_fill(~valid_mask, float("-inf"))

        # ── Add epsilon (no antecedent) score = 0 ─────────────────────────
        epsilon = torch.zeros(batch, num_spans, 1, device=device)
        antecedent_scores = torch.cat([epsilon, total_scores], dim=-1)

        return antecedent_scores, top_k_idx, valid_mask


# ══════════════════════════════════════════════════════════════════════════════
# Coreference Head
# ══════════════════════════════════════════════════════════════════════════════

class CoreferenceHead(nn.Module):
    """
    End-to-end Arabic Coreference Resolution Head.
    Implements Lee et al. (2018) adapted for Arabic.

    Plug-in interface matches NERHead and POSHead for MTL compatibility:
      forward(sequence_output, attention_mask, clusters=None) →
          {"loss", "predictions", "mention_scores"}

    Args:
        hidden_size:        Backbone output dimension.
        span_width_emb_dim: Width embedding dimension.
        max_span_width:     Maximum mention span width in tokens.
        top_lambda:         Fraction of spans kept after mention pruning
                            (Lee et al. use 0.4).
        antecedent_k:       Max antecedent candidates per span.
        mention_hidden:     Hidden size of mention scorer.
        antecedent_hidden:  Hidden size of antecedent scorer.
        dropout:            Dropout probability.
    """

    def __init__(
        self,
        hidden_size:        int   = 768,
        span_width_emb_dim: int   = 64,
        max_span_width:     int   = 30,
        top_lambda:         float = 0.4,
        antecedent_k:       int   = 50,
        mention_hidden:     int   = 150,
        antecedent_hidden:  int   = 150,
        dropout:            float = 0.3,
    ) -> None:
        super().__init__()

        self.max_span_width = max_span_width
        self.top_lambda     = top_lambda
        self.antecedent_k   = antecedent_k

        # ── Span representation module ────────────────────────────────────
        self.span_repr = SpanRepresentation(
            hidden_size=hidden_size,
            span_width_emb_dim=span_width_emb_dim,
            max_span_width=max_span_width,
            dropout=dropout,
        )
        span_dim = self.span_repr.output_dim   # 3H + W

        # ── Mention scorer ────────────────────────────────────────────────
        self.mention_scorer = MentionScorer(
            input_dim=span_dim,
            hidden_dim=mention_hidden,
            dropout=dropout,
        )

        # ── Antecedent scorer ─────────────────────────────────────────────
        self.antecedent_scorer = AntecedentScorer(
            input_dim=span_dim,
            hidden_dim=antecedent_hidden,
            dropout=dropout,
        )

    # ── Span candidate generation ─────────────────────────────────────────
    def _generate_spans(
        self,
        seq_len: int,
        device:  torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate all candidate spans up to max_span_width.
        Returns (starts, ends) each of shape (num_candidates,).
        """
        starts, ends = [], []
        for start in range(seq_len):
            for width in range(self.max_span_width):
                end = start + width
                if end < seq_len:
                    starts.append(start)
                    ends.append(end)
        return (
            torch.tensor(starts, device=device),
            torch.tensor(ends,   device=device),
        )

    # ── Loss computation ──────────────────────────────────────────────────
    def _compute_loss(
        self,
        antecedent_scores: torch.Tensor,   # (batch, num_spans, k+1)
        top_k_indices:     torch.Tensor,   # (batch, num_spans, k)
        gold_clusters:     list,           # list of cluster lists per example
        span_starts:       torch.Tensor,   # (batch, num_spans)
        span_ends:         torch.Tensor,   # (batch, num_spans)
    ) -> torch.Tensor:
        """
        Compute mention-ranking loss (Lee et al. 2018).
        For each span, the gold antecedent set is all previous spans
        in the same coreference cluster.
        """
        batch, num_spans, num_ant = antecedent_scores.shape
        device = antecedent_scores.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0

        for b in range(batch):
            clusters = gold_clusters[b] if b < len(gold_clusters) else []

            # Build mention-to-cluster mapping from gold clusters
            mention_to_cluster: dict[tuple, int] = {}
            for cid, cluster in enumerate(clusters):
                for mention in cluster:
                    mention_to_cluster[tuple(mention)] = cid

            for i in range(num_spans):
                start_i = span_starts[b, i].item()
                end_i   = span_ends[b, i].item()
                cluster_i = mention_to_cluster.get((start_i, end_i), -1)

                # Gold antecedents: previous spans in the same cluster
                gold_ant = []
                for j_idx in range(num_ant - 1):  # skip epsilon
                    j = top_k_indices[b, i, j_idx].item()
                    if j >= i:
                        continue
                    start_j = span_starts[b, j].item()
                    end_j   = span_ends[b, j].item()
                    if cluster_i >= 0 and \
                       mention_to_cluster.get((start_j, end_j), -1) == cluster_i:
                        gold_ant.append(j_idx + 1)  # +1 for epsilon offset

                # If no gold antecedent → epsilon (index 0) is gold
                if not gold_ant:
                    gold_ant = [0]

                # Mention-ranking loss: log-sum-exp over gold antecedents
                gold_scores = antecedent_scores[b, i, gold_ant]
                all_scores  = antecedent_scores[b, i]
                log_norm    = torch.logsumexp(all_scores, dim=0)
                log_gold    = torch.logsumexp(gold_scores, dim=0)
                total_loss  = total_loss + (log_norm - log_gold)
                count += 1

        return total_loss / max(count, 1)

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        sequence_output: torch.Tensor,              # (B, seq_len, H)
        attention_mask:  torch.Tensor,              # (B, seq_len)
        clusters:        Optional[list] = None,     # gold clusters per example
    ) -> dict:
        """
        Args:
            sequence_output: Subword-level hidden states from backbone.
            attention_mask:  Subword-level mask.
            clusters:        Gold coreference clusters (list of lists of
                             [start, end] pairs). Required for training.

        Returns:
            dict with keys:
              "loss"          — mention-ranking loss (if clusters provided)
              "mention_scores"— raw mention scores (batch, num_spans)
              "predictions"   — list of predicted cluster sets per example
              "span_starts"   — candidate span starts (batch, num_spans)
              "span_ends"     — candidate span ends (batch, num_spans)
        """
        batch, seq_len, _ = sequence_output.shape
        device = sequence_output.device

        # ── 1. Generate candidate spans ───────────────────────────────────
        flat_starts, flat_ends = self._generate_spans(seq_len, device)
        num_candidates = flat_starts.shape[0]

        span_starts = flat_starts.unsqueeze(0).expand(batch, -1)   # (B, C)
        span_ends   = flat_ends.unsqueeze(0).expand(batch, -1)     # (B, C)

        # ── 2. Compute span representations ──────────────────────────────
        span_repr = self.span_repr(sequence_output, span_starts, span_ends)

        # ── 3. Score mentions ─────────────────────────────────────────────
        mention_scores = self.mention_scorer(span_repr)             # (B, C)

        # ── 4. Prune to top-λT spans ──────────────────────────────────────
        num_keep = max(1, int(self.top_lambda * seq_len))
        num_keep = min(num_keep, num_candidates)

        _, top_indices = mention_scores.topk(num_keep, dim=1)      # (B, num_keep)
        top_indices, _ = top_indices.sort(dim=1)                   # maintain order

        # Gather pruned spans
        top_starts  = torch.gather(span_starts, 1, top_indices)    # (B, num_keep)
        top_ends    = torch.gather(span_ends,   1, top_indices)
        top_repr    = batch_select(span_repr, top_indices)          # (B, num_keep, D)
        top_scores  = torch.gather(mention_scores, 1, top_indices)

        # ── 5. Score antecedents ──────────────────────────────────────────
        k = min(self.antecedent_k, num_keep)
        antecedent_scores, top_k_idx, ant_mask = self.antecedent_scorer(
            top_repr, top_scores, top_k=k
        )

        # ── 6. Compute loss (training only) ───────────────────────────────
        output: dict = {
            "mention_scores": mention_scores,
            "span_starts":    top_starts,
            "span_ends":      top_ends,
        }

        if clusters is not None:
            loss = self._compute_loss(
                antecedent_scores, top_k_idx,
                clusters, top_starts, top_ends,
            )
            output["loss"] = loss

        # ── 7. Predict clusters (greedy highest-scoring antecedent) ───────
        output["predictions"] = self._predict_clusters(
            antecedent_scores, top_k_idx,
            top_starts, top_ends, batch,
        )

        return output

    # ── Cluster prediction ────────────────────────────────────────────────
    def _predict_clusters(
        self,
        antecedent_scores: torch.Tensor,   # (B, num_spans, k+1)
        top_k_indices:     torch.Tensor,   # (B, num_spans, k)
        span_starts:       torch.Tensor,
        span_ends:         torch.Tensor,
        batch:             int,
    ) -> list[list[list]]:
        """
        Greedy decoding: each span links to its highest-scoring antecedent.
        Transitively merges into coreference clusters.
        """
        all_clusters = []

        for b in range(batch):
            # Union-Find for cluster merging
            parent: dict[int, int] = {}

            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                parent[find(x)] = find(y)

            num_spans = antecedent_scores.shape[1]
            scores_b  = antecedent_scores[b]        # (num_spans, k+1)
            best_ant  = scores_b.argmax(dim=-1)     # (num_spans,)

            for i in range(num_spans):
                ant_idx = best_ant[i].item()
                if ant_idx == 0:
                    continue   # epsilon — no antecedent
                j = top_k_indices[b, i, ant_idx - 1].item()
                if j < i:
                    union(i, j)

            # Group spans by cluster root
            cluster_map: dict[int, list] = {}
            for i in range(num_spans):
                root = find(i)
                mention = [
                    span_starts[b, i].item(),
                    span_ends[b, i].item(),
                ]
                cluster_map.setdefault(root, []).append(mention)

            # Only keep clusters with 2+ mentions
            clusters = [
                sorted(spans) for spans in cluster_map.values()
                if len(spans) >= 2
            ]
            all_clusters.append(clusters)

        return all_clusters


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Coreference Head Sanity Check ===\n")

    B, S, H = 2, 30, 768

    head = CoreferenceHead(
        hidden_size=H,
        max_span_width=5,
        top_lambda=0.4,
        antecedent_k=10,
    )
    head.eval()

    seq_out = torch.randn(B, S, H)
    mask    = torch.ones(B, S, dtype=torch.long)
    mask[1, 20:] = 0

    # Dummy gold clusters: [[mention1, mention2], ...]
    clusters = [
        [[[0, 1], [5, 6]], [[10, 10], [15, 16]]],   # batch 0: 2 clusters
        [[[0, 0], [3, 3]]],                           # batch 1: 1 cluster
    ]

    with torch.no_grad():
        # Inference (no clusters)
        out_infer = head(seq_out, mask)
        assert "mention_scores" in out_infer
        assert "predictions"    in out_infer
        assert "span_starts"    in out_infer
        print(f"✓ Inference: {len(out_infer['predictions'])} examples decoded")
        print(f"  mention_scores shape : {out_infer['mention_scores'].shape}")
        print(f"  span_starts shape    : {out_infer['span_starts'].shape}")
        n_clusters = sum(len(c) for c in out_infer["predictions"])
        print(f"  predicted clusters   : {n_clusters} total")

        # Training (with clusters)
        out_train = head(seq_out, mask, clusters=clusters)
        assert "loss" in out_train
        print(f"\n✓ Training: loss={out_train['loss'].item():.4f}")

    # ── SpanRepresentation standalone ────────────────────────────────────
    span_repr = SpanRepresentation(hidden_size=H, max_span_width=5)
    starts = torch.tensor([[0, 1, 5], [0, 2, 8]])
    ends   = torch.tensor([[1, 3, 7], [0, 4, 9]])
    repr_out = span_repr(seq_out, starts, ends)
    assert repr_out.shape == (B, 3, span_repr.output_dim)
    print(f"\n✓ SpanRepresentation: output shape {repr_out.shape}")
    print(f"  span_dim = 3×{H} + {span_repr.span_width_emb_dim} "
          f"= {span_repr.output_dim}")

    # ── MentionScorer standalone ──────────────────────────────────────────
    mention_scorer = MentionScorer(input_dim=span_repr.output_dim)
    scores = mention_scorer(repr_out)
    assert scores.shape == (B, 3)
    print(f"\n✓ MentionScorer: output shape {scores.shape}")

    print("\n✅ All checks passed — src/models/coref_head.py ready")
