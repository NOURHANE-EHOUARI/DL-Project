"""
coref_eval.py — Coreference Resolution Evaluation:
               MUC, B-cubed, CEAF-e, CoNLL Average F1.

Owner:   Student B (Hiba)
Phase:   4 — Evaluation & Analysis (Week 5–6)

Metrics computed:
    - MUC  : Mention-based F1 (Vilain et al. 1995)
    - B³   : Entity-level F1 (Bagga & Baldwin 1998)
    - CEAFe: Entity alignment F1 (Luo 2005)
    - CoNLL Average F1 = (MUC + B³ + CEAFe) / 3  (standard benchmark metric)
    - Mention recall / precision
    - Cluster-level qualitative analysis
    - Bootstrap confidence intervals

References:
    - Pradhan et al. (2012) CoNLL Shared Task scorer
    - Lee et al. (2018) End-to-end coreference
    - Inoue et al. (2020) Arabic coreference

Note: This is a pure-Python implementation of the standard
      CoNLL coreference metrics, compatible with our CoNLL placeholder
      data. For production use, the official Perl scorer can be called
      via subprocess — see score_with_official_scorer().
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Type aliases
# Cluster  = frozenset of mention tuples, each mention = (start, end)
# Clusters = list of Cluster
# ══════════════════════════════════════════════════════════════════════════════

Mention  = tuple[int, int]          # (start_token, end_token) inclusive
Cluster  = frozenset                # frozenset[Mention]
Clusters = list[Cluster]


def to_clusters(raw: list[list[list[int]]]) -> Clusters:
    """
    Convert raw cluster format from datasets.py / coref_head.py to
    frozenset format used by the evaluator.

    Input:  [[[start, end], [start, end]], [[start, end], ...]]
    Output: [frozenset({(start,end), ...}), ...]
    """
    return [
        frozenset((m[0], m[1]) for m in cluster)
        for cluster in raw
        if len(cluster) >= 2
    ]


def mention_set(clusters: Clusters) -> set[Mention]:
    """All mentions across all clusters."""
    return {m for cluster in clusters for m in cluster}


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricScore:
    """P / R / F1 triple for one metric."""
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0

    def __str__(self) -> str:
        return (f"P={self.precision*100:.2f}%  "
                f"R={self.recall*100:.2f}%  "
                f"F1={self.f1*100:.2f}%")

    @staticmethod
    def from_prf(p: float, r: float) -> "MetricScore":
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return MetricScore(precision=p, recall=r, f1=f1)


@dataclass
class CorefResult:
    """Holds all evaluation outputs for one coreference experiment."""

    muc:   MetricScore = field(default_factory=MetricScore)
    b3:    MetricScore = field(default_factory=MetricScore)
    ceafe: MetricScore = field(default_factory=MetricScore)

    # CoNLL average = (MUC F1 + B³ F1 + CEAFe F1) / 3
    conll_f1: float = 0.0

    # Confidence interval on CoNLL F1
    conll_ci_low:  float = 0.0
    conll_ci_high: float = 0.0

    # Mention-level stats
    mention_precision: float = 0.0
    mention_recall:    float = 0.0
    mention_f1:        float = 0.0

    # Cluster stats
    n_gold_clusters: int = 0
    n_pred_clusters: int = 0
    n_gold_mentions: int = 0
    n_pred_mentions: int = 0

    # Qualitative cluster analysis
    cluster_analysis: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  COREFERENCE RESOLUTION EVALUATION SUMMARY",
            "=" * 60,
            f"  MUC    : {self.muc}",
            f"  B³     : {self.b3}",
            f"  CEAFe  : {self.ceafe}",
            "  " + "─" * 50,
            f"  CoNLL Avg F1 : {self.conll_f1*100:.2f}%"
            f"  [{self.conll_ci_low*100:.2f} – {self.conll_ci_high*100:.2f}]",
            "",
            f"  Mention P : {self.mention_precision*100:.2f}%",
            f"  Mention R : {self.mention_recall*100:.2f}%",
            f"  Mention F1: {self.mention_f1*100:.2f}%",
            "",
            f"  Gold clusters : {self.n_gold_clusters}",
            f"  Pred clusters : {self.n_pred_clusters}",
            f"  Gold mentions : {self.n_gold_mentions}",
            f"  Pred mentions : {self.n_pred_mentions}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MUC metric (Vilain et al. 1995)
# ══════════════════════════════════════════════════════════════════════════════

def muc_score(
    gold_clusters: Clusters,
    pred_clusters: Clusters,
) -> MetricScore:
    """
    MUC coreference metric.

    Recall    = Σ_k (|K_k| - |p(K_k)|) / Σ_k (|K_k| - 1)
    Precision = Σ_k (|R_k| - |p(R_k)|) / Σ_k (|R_k| - 1)

    where p(S) = number of predicted (gold) clusters that partition S.

    Args:
        gold_clusters: Gold coreference clusters.
        pred_clusters: Predicted coreference clusters.

    Returns:
        MetricScore with MUC P / R / F1.
    """
    def _partition_count(key_cluster: Cluster, response: Clusters) -> int:
        """Number of response clusters that intersect key_cluster."""
        return sum(
            1 for r in response if key_cluster & r
        )

    def _recall_numerator(key: Clusters, response: Clusters) -> float:
        return sum(
            len(k) - _partition_count(k, response)
            for k in key
            if len(k) > 1
        )

    def _denom(clusters: Clusters) -> int:
        return sum(len(k) - 1 for k in clusters if len(k) > 1)

    recall_num  = _recall_numerator(gold_clusters, pred_clusters)
    recall_den  = _denom(gold_clusters)
    prec_num    = _recall_numerator(pred_clusters, gold_clusters)
    prec_den    = _denom(pred_clusters)

    r = recall_num / recall_den if recall_den > 0 else 0.0
    p = prec_num   / prec_den   if prec_den   > 0 else 0.0

    return MetricScore.from_prf(p, r)


# ══════════════════════════════════════════════════════════════════════════════
# B-cubed metric (Bagga & Baldwin 1998)
# ══════════════════════════════════════════════════════════════════════════════

def b3_score(
    gold_clusters: Clusters,
    pred_clusters: Clusters,
) -> MetricScore:
    """
    B-cubed coreference metric.

    For each mention m:
      Recall(m)    = |K(m) ∩ R(m)| / |K(m)|
      Precision(m) = |K(m) ∩ R(m)| / |R(m)|

    Overall P/R = average over all mentions.

    Args:
        gold_clusters: Gold coreference clusters.
        pred_clusters: Predicted coreference clusters.

    Returns:
        MetricScore with B³ P / R / F1.
    """
    # Build mention → cluster maps
    gold_map: dict[Mention, Cluster] = {}
    for cluster in gold_clusters:
        for mention in cluster:
            gold_map[mention] = cluster

    pred_map: dict[Mention, Cluster] = {}
    for cluster in pred_clusters:
        for mention in cluster:
            pred_map[mention] = cluster

    all_mentions = set(gold_map.keys()) | set(pred_map.keys())
    if not all_mentions:
        return MetricScore()

    recall_sum, prec_sum = 0.0, 0.0

    for mention in all_mentions:
        gold_c = gold_map.get(mention, frozenset({mention}))
        pred_c = pred_map.get(mention, frozenset({mention}))

        intersection = len(gold_c & pred_c)

        recall_sum += intersection / len(gold_c) if gold_c else 0.0
        prec_sum   += intersection / len(pred_c) if pred_c else 0.0

    n = len(all_mentions)
    r = recall_sum / n
    p = prec_sum   / n

    return MetricScore.from_prf(p, r)


# ══════════════════════════════════════════════════════════════════════════════
# CEAF-e metric (Luo 2005)
# ══════════════════════════════════════════════════════════════════════════════

def ceafe_score(
    gold_clusters: Clusters,
    pred_clusters: Clusters,
) -> MetricScore:
    """
    CEAF-e (Entity-level CEAF) coreference metric.

    Finds the best one-to-one alignment between gold and predicted clusters
    using the Hungarian algorithm (or greedy approximation for speed).

    Similarity φ_4(K, R) = |K ∩ R| / (|K| + |R| - |K ∩ R|)  (Jaccard)

    Args:
        gold_clusters: Gold coreference clusters.
        pred_clusters: Predicted coreference clusters.

    Returns:
        MetricScore with CEAFe P / R / F1.
    """
    if not gold_clusters or not pred_clusters:
        return MetricScore()

    def phi4(k: Cluster, r: Cluster) -> float:
        intersection = len(k & r)
        union        = len(k | r)
        return intersection / union if union > 0 else 0.0

    # Build similarity matrix
    n_gold = len(gold_clusters)
    n_pred = len(pred_clusters)
    sim    = np.zeros((n_gold, n_pred))

    for i, g in enumerate(gold_clusters):
        for j, p in enumerate(pred_clusters):
            sim[i, j] = phi4(g, p)

    # Greedy alignment (sufficient for our evaluation)
    # For exact results, use scipy.optimize.linear_sum_assignment
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-sim)
        total_sim = sim[row_ind, col_ind].sum()
    except ImportError:
        # Greedy fallback
        used_pred = set()
        total_sim = 0.0
        for i in range(n_gold):
            best_j, best_s = -1, -1.0
            for j in range(n_pred):
                if j not in used_pred and sim[i, j] > best_s:
                    best_j, best_s = j, sim[i, j]
            if best_j >= 0:
                used_pred.add(best_j)
                total_sim += best_s

    p = total_sim / n_pred if n_pred > 0 else 0.0
    r = total_sim / n_gold if n_gold > 0 else 0.0

    return MetricScore.from_prf(p, r)


# ══════════════════════════════════════════════════════════════════════════════
# Mention-level metrics
# ══════════════════════════════════════════════════════════════════════════════

def mention_score(
    gold_clusters: Clusters,
    pred_clusters: Clusters,
) -> MetricScore:
    """
    Mention detection P / R / F1.
    Measures how well the system identifies mention boundaries,
    independent of clustering.
    """
    gold_mentions = mention_set(gold_clusters)
    pred_mentions = mention_set(pred_clusters)

    tp = len(gold_mentions & pred_mentions)
    fp = len(pred_mentions - gold_mentions)
    fn = len(gold_mentions - pred_mentions)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return MetricScore.from_prf(p, r)


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluator
# ══════════════════════════════════════════════════════════════════════════════

class CorefEvaluator:
    """
    Evaluates coreference resolution predictions against gold clusters.

    Inputs are lists of documents, each document being a list of clusters.
    Each cluster is a list of [start, end] mention spans.

    Args:
        gold_docs: Gold clusters per document.
                   Format: list of list of list of [start, end]
        pred_docs: Predicted clusters per document (same format).
        tokens:    Optional token lists per document for qualitative analysis.
    """

    def __init__(
        self,
        gold_docs: list[list[list[list[int]]]],
        pred_docs: list[list[list[list[int]]]],
        tokens:    list[list[str]] | None = None,
    ) -> None:
        assert len(gold_docs) == len(pred_docs), \
            f"Length mismatch: {len(gold_docs)} gold vs {len(pred_docs)} pred"
        self.gold_docs = [to_clusters(d) for d in gold_docs]
        self.pred_docs = [to_clusters(d) for d in pred_docs]
        self.tokens    = tokens or [[] for _ in gold_docs]

    # ── Main entry point ──────────────────────────────────────────────────
    def evaluate(
        self,
        bootstrap_n: int = 500,
        seed:        int = 42,
    ) -> CorefResult:
        """
        Run full coreference evaluation suite.

        Args:
            bootstrap_n: Bootstrap iterations for CI (0 = skip).
            seed:        Random seed.

        Returns:
            CorefResult with all metrics populated.
        """
        result = CorefResult()

        # ── Aggregate clusters across all documents ───────────────────────
        all_gold = []
        all_pred = []
        for g_doc, p_doc in zip(self.gold_docs, self.pred_docs):
            all_gold.extend(g_doc)
            all_pred.extend(p_doc)

        # ── Core metrics ──────────────────────────────────────────────────
        result.muc   = muc_score(all_gold, all_pred)
        result.b3    = b3_score(all_gold,  all_pred)
        result.ceafe = ceafe_score(all_gold, all_pred)

        result.conll_f1 = (
            result.muc.f1 + result.b3.f1 + result.ceafe.f1
        ) / 3.0

        # ── Mention-level stats ───────────────────────────────────────────
        ms = mention_score(all_gold, all_pred)
        result.mention_precision = ms.precision
        result.mention_recall    = ms.recall
        result.mention_f1        = ms.f1

        # ── Cluster counts ────────────────────────────────────────────────
        result.n_gold_clusters = len(all_gold)
        result.n_pred_clusters = len(all_pred)
        result.n_gold_mentions = len(mention_set(all_gold))
        result.n_pred_mentions = len(mention_set(all_pred))

        # ── Bootstrap CI on CoNLL F1 ──────────────────────────────────────
        if bootstrap_n > 0:
            ci_low, ci_high = self._bootstrap_ci(bootstrap_n, seed)
            result.conll_ci_low  = ci_low
            result.conll_ci_high = ci_high

        # ── Qualitative cluster analysis ──────────────────────────────────
        result.cluster_analysis = self._cluster_analysis()

        return result

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    def _bootstrap_ci(
        self, n: int = 500, seed: int = 42, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Bootstrap CI on CoNLL average F1."""
        rng     = random.Random(seed)
        indices = list(range(len(self.gold_docs)))
        scores  = []

        for _ in range(n):
            sample = rng.choices(indices, k=len(indices))
            g_all, p_all = [], []
            for i in sample:
                g_all.extend(self.gold_docs[i])
                p_all.extend(self.pred_docs[i])

            muc_f   = muc_score(g_all, p_all).f1
            b3_f    = b3_score(g_all, p_all).f1
            ceafe_f = ceafe_score(g_all, p_all).f1
            scores.append((muc_f + b3_f + ceafe_f) / 3.0)

        scores.sort()
        lo = int(alpha / 2 * n)
        hi = int((1 - alpha / 2) * n)
        return scores[lo], scores[hi]

    # ── Qualitative analysis ──────────────────────────────────────────────
    def _cluster_analysis(self) -> list[dict]:
        """
        Per-document cluster quality analysis.
        Identifies singleton errors, merged clusters, and split clusters.
        Useful for the jury's error analysis slide.
        """
        analysis = []

        for doc_idx, (gold_clusters, pred_clusters) in enumerate(
            zip(self.gold_docs, self.pred_docs)
        ):
            gold_mentions = mention_set(gold_clusters)
            pred_mentions = mention_set(pred_clusters)

            # Singleton errors: predicted as singleton but in gold cluster
            singletons = []
            for m in gold_mentions - pred_mentions:
                singletons.append(m)

            # Merged clusters: two gold clusters merged into one predicted
            merged = 0
            for p_cluster in pred_clusters:
                gold_ids = set()
                for g_idx, g_cluster in enumerate(gold_clusters):
                    if p_cluster & g_cluster:
                        gold_ids.add(g_idx)
                if len(gold_ids) > 1:
                    merged += 1

            # Split clusters: one gold cluster split into multiple predicted
            split = 0
            for g_cluster in gold_clusters:
                pred_ids = set()
                for p_idx, p_cluster in enumerate(pred_clusters):
                    if g_cluster & p_cluster:
                        pred_ids.add(p_idx)
                if len(pred_ids) > 1:
                    split += 1

            tokens = self.tokens[doc_idx] if doc_idx < len(self.tokens) else []
            analysis.append({
                "doc_idx":         doc_idx,
                "gold_clusters":   len(gold_clusters),
                "pred_clusters":   len(pred_clusters),
                "singleton_errors":len(singletons),
                "merged_clusters": merged,
                "split_clusters":  split,
                "sample_mention":  (
                    [tokens[m[0]:m[1]+1] for m in list(singletons)[:3]]
                    if tokens else singletons[:3]
                ),
            })

        return analysis


# ══════════════════════════════════════════════════════════════════════════════
# W&B logging
# ══════════════════════════════════════════════════════════════════════════════

def log_to_wandb(result: CorefResult, step: int | None = None) -> None:
    """Log CorefResult metrics to Weights & Biases."""
    try:
        import wandb
        metrics = {
            "eval/coref_muc_p":         result.muc.precision,
            "eval/coref_muc_r":         result.muc.recall,
            "eval/coref_muc_f1":        result.muc.f1,
            "eval/coref_b3_p":          result.b3.precision,
            "eval/coref_b3_r":          result.b3.recall,
            "eval/coref_b3_f1":         result.b3.f1,
            "eval/coref_ceafe_p":       result.ceafe.precision,
            "eval/coref_ceafe_r":       result.ceafe.recall,
            "eval/coref_ceafe_f1":      result.ceafe.f1,
            "eval/coref_conll_f1":      result.conll_f1,
            "eval/coref_conll_ci_low":  result.conll_ci_low,
            "eval/coref_conll_ci_high": result.conll_ci_high,
            "eval/coref_mention_p":     result.mention_precision,
            "eval/coref_mention_r":     result.mention_recall,
            "eval/coref_mention_f1":    result.mention_f1,
        }
        wandb.log(metrics, step=step)
        print("✓ Coref metrics logged to W&B")
    except ImportError:
        print("⚠  wandb not available — skipping W&B logging")


# ══════════════════════════════════════════════════════════════════════════════
# Save report
# ══════════════════════════════════════════════════════════════════════════════

def save_coref_report(
    result: CorefResult,
    path: str | Path = "coref_eval_report.json",
) -> None:
    """Save full coreference evaluation report to JSON."""
    report = {
        "muc":   {"p": result.muc.precision,
                  "r": result.muc.recall,
                  "f1": result.muc.f1},
        "b3":    {"p": result.b3.precision,
                  "r": result.b3.recall,
                  "f1": result.b3.f1},
        "ceafe": {"p": result.ceafe.precision,
                  "r": result.ceafe.recall,
                  "f1": result.ceafe.f1},
        "conll_f1":      result.conll_f1,
        "conll_ci_low":  result.conll_ci_low,
        "conll_ci_high": result.conll_ci_high,
        "mention_p":     result.mention_precision,
        "mention_r":     result.mention_recall,
        "mention_f1":    result.mention_f1,
        "n_gold_clusters": result.n_gold_clusters,
        "n_pred_clusters": result.n_pred_clusters,
        "n_gold_mentions": result.n_gold_mentions,
        "n_pred_mentions": result.n_pred_mentions,
        "cluster_analysis": result.cluster_analysis[:20],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✓ Coref eval report saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Coreference Evaluation Sanity Check ===\n")

    # Toy data — 2 documents
    # Format: list of clusters, each cluster is list of [start, end] mentions
    gold_docs = [
        [
            [[0, 0], [5, 5], [10, 11]],   # cluster 1: 3 mentions
            [[2, 3], [7, 8]],              # cluster 2: 2 mentions
        ],
        [
            [[0, 1], [4, 4]],              # cluster 1: 2 mentions
            [[2, 2], [6, 7], [9, 9]],      # cluster 2: 3 mentions
        ],
    ]

    # Simulate imperfect predictions
    pred_docs = [
        [
            [[0, 0], [5, 5]],              # missed [10,11]
            [[2, 3], [7, 8], [10, 11]],    # merged with wrong cluster
        ],
        [
            [[0, 1], [4, 4], [2, 2]],      # merged clusters 1+2 partially
            [[6, 7], [9, 9]],              # split cluster 2
        ],
    ]

    tokens = [
        ["محمد", "علي", "زار", "الرئيس", "أحمد",
         "هو", "يعمل", "في", "الشركة", "كما",
         "المدير", "العام"],
        ["الفريق", "الوطني", "فاز", "بعد",
         "هم", "يلعبون", "في", "الملعب",
         "الكبير", "الآن"],
    ]

    evaluator = CorefEvaluator(gold_docs, pred_docs, tokens)
    result    = evaluator.evaluate(bootstrap_n=200, seed=42)

    print(result.summary())

    # Assertions
    assert 0.0 <= result.muc.f1   <= 1.0, "MUC F1 out of range"
    assert 0.0 <= result.b3.f1    <= 1.0, "B³ F1 out of range"
    assert 0.0 <= result.ceafe.f1 <= 1.0, "CEAFe F1 out of range"
    assert 0.0 <= result.conll_f1 <= 1.0, "CoNLL F1 out of range"
    assert result.conll_ci_low <= result.conll_f1 <= result.conll_ci_high
    assert result.n_gold_clusters == 4,   "Should have 4 gold clusters"
    assert result.n_gold_mentions == 10,  "Should have 10 gold mentions"

    print(f"\n  MUC  : {result.muc}")
    print(f"  B³   : {result.b3}")
    print(f"  CEAFe: {result.ceafe}")
    print(f"\n  CoNLL Avg F1: {result.conll_f1*100:.2f}%")

    print("\n  Cluster analysis per document:")
    for doc in result.cluster_analysis:
        print(f"    doc {doc['doc_idx']}: "
              f"gold={doc['gold_clusters']} pred={doc['pred_clusters']} "
              f"singletons={doc['singleton_errors']} "
              f"merged={doc['merged_clusters']} "
              f"split={doc['split_clusters']}")

    print("\n  to_clusters() test:")
    raw = [[[0, 1], [4, 5]], [[2, 2], [7, 8]]]
    clusters = to_clusters(raw)
    assert len(clusters) == 2
    assert frozenset([(0, 1), (4, 5)]) in clusters
    print(f"    {raw} → {[set(c) for c in clusters]}")

    print("\n✅ All checks passed — src/evaluation/coref_eval.py ready")
