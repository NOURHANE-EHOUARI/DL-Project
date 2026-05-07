"""
pos_eval.py — POS Tagging Evaluation: Token Accuracy, Per-tag Breakdown,
              Confusion Matrix, Error Analysis, Dialect Robustness.

Owner:   Student B (Hiba)
Phase:   4 — Evaluation & Analysis (Week 5–6)

Metrics computed:
    - Overall token accuracy (excluding padding)
    - Per-tag accuracy and support
    - Macro / weighted F1 over POS tags
    - Confusion matrix (top confused tag pairs)
    - Error analysis by morphological category
    - Dialect-specific accuracy breakdown (Egyptian, Gulf, Levantine, Maghrebi)
    - Bootstrap confidence intervals
    - Comparison against Farasa baseline

Dependencies:
    pip install scikit-learn pandas matplotlib seaborn
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class POSResult:
    """Holds all evaluation outputs for one POS experiment run."""

    # Overall metrics
    accuracy:        float = 0.0
    macro_f1:        float = 0.0
    weighted_f1:     float = 0.0

    # Confidence interval on accuracy
    acc_ci_low:  float = 0.0
    acc_ci_high: float = 0.0

    # Per-tag metrics  {tag: {"accuracy": float, "f1": float, "support": int}}
    per_tag: dict = field(default_factory=dict)

    # Top confused pairs  [(gold_tag, pred_tag, count), ...]
    top_confusions: list = field(default_factory=list)

    # Error counts
    n_correct:    int = 0
    n_incorrect:  int = 0
    n_total:      int = 0

    # Dialect breakdown  {dialect: accuracy}
    dialect_accuracy: dict = field(default_factory=dict)

    # Morphological category breakdown  {category: accuracy}
    morph_accuracy: dict = field(default_factory=dict)

    # Raw errors  list[dict]
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 58,
            "  POS TAGGING EVALUATION SUMMARY",
            "=" * 58,
            f"  Token Accuracy : {self.accuracy * 100:.2f}%"
            f"  [{self.acc_ci_low*100:.2f} – {self.acc_ci_high*100:.2f}]",
            f"  Macro F1       : {self.macro_f1 * 100:.2f}%",
            f"  Weighted F1    : {self.weighted_f1 * 100:.2f}%",
            f"  Correct        : {self.n_correct:,} / {self.n_total:,} tokens",
            "",
        ]

        if self.dialect_accuracy:
            lines.append("  Dialect Breakdown:")
            for dialect, acc in sorted(self.dialect_accuracy.items()):
                lines.append(f"    {dialect:<12}: {acc*100:.2f}%")
            lines.append("")

        lines.append("  Top-15 POS Tags by F1:")
        for tag, m in sorted(
            self.per_tag.items(), key=lambda x: -x[1]["f1"]
        )[:15]:
            lines.append(
                f"    {tag:<28}  Acc={m['accuracy']*100:.1f}%"
                f"  F1={m['f1']*100:.1f}%  (n={m['support']})"
            )

        if self.top_confusions:
            lines.append("\n  Top-10 Confusion Pairs (gold → pred):")
            for gold, pred, count in self.top_confusions[:10]:
                lines.append(f"    {gold:<20} → {pred:<20}  ({count}×)")

        lines.append("=" * 58)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluator
# ══════════════════════════════════════════════════════════════════════════════

class POSEvaluator:
    """
    Evaluates POS tagging predictions against gold labels.

    All inputs are lists of sentences, each a list of POS tag strings.

    Args:
        labels:       Gold POS tag sequences.
        predictions:  Predicted POS tag sequences.
        tokens:       Optional word strings for error analysis.
        dialects:     Optional dialect label per sentence
                      ('egy', 'glf', 'lev', 'mgr').
    """

    # Morphological categories for Arabic POS error analysis
    MORPH_CATEGORIES = {
        "verb":       {"V", "FUT_PART+V", "PROG_PART+V",
                       "FUT_PART+V+PRON", "PROG_PART+V+PRON", "CONJ+V"},
        "noun":       {"NOUN", "DET+NOUN", "CONJ+NOUN", "NOUN+PRON",
                       "NOUN+PREP+PRON"},
        "clitic":     {"PREP+PRON", "CONJ+PART", "CONJ+PRON",
                       "PART+PRON", "PREP+DET"},
        "adjective":  {"ADJ", "CONJ+ADJ", "ADJ+PRON"},
        "pronoun":    {"PRON", "V+PRON"},
        "particle":   {"PART", "DET", "PREP", "CONJ", "ADV"},
        "other":      {"NUM", "PUNC", "ABBREV", "FOREIGN", "OTHER"},
    }

    def __init__(
        self,
        labels:      list[list[str]],
        predictions: list[list[str]],
        tokens:      list[list[str]] | None = None,
        dialects:    list[str] | None = None,
    ) -> None:
        assert len(labels) == len(predictions), \
            f"Length mismatch: {len(labels)} gold vs {len(predictions)} pred"
        self.labels      = labels
        self.predictions = predictions
        self.tokens      = tokens or [[] for _ in labels]
        self.dialects    = dialects or []

    # ── Main entry point ──────────────────────────────────────────────────
    def evaluate(
        self,
        bootstrap_n: int = 1000,
        seed:        int = 42,
        top_k_conf:  int = 20,
    ) -> POSResult:
        """
        Run full POS evaluation suite.

        Args:
            bootstrap_n: Bootstrap iterations for CI (0 = skip).
            seed:        Random seed.
            top_k_conf:  Number of top confused pairs to report.

        Returns:
            POSResult with all metrics populated.
        """
        result = POSResult()

        # ── Flatten sequences (ignoring padding) ──────────────────────────
        flat_gold, flat_pred, flat_tok = self._flatten()
        result.n_total   = len(flat_gold)
        result.n_correct = sum(g == p for g, p in zip(flat_gold, flat_pred))
        result.n_incorrect = result.n_total - result.n_correct
        result.accuracy  = result.n_correct / result.n_total if result.n_total else 0.0

        # ── Sklearn metrics ───────────────────────────────────────────────
        try:
            from sklearn.metrics import f1_score
            all_tags = sorted(set(flat_gold + flat_pred))
            result.macro_f1    = f1_score(flat_gold, flat_pred,
                                          average="macro",
                                          zero_division=0,
                                          labels=all_tags)
            result.weighted_f1 = f1_score(flat_gold, flat_pred,
                                          average="weighted",
                                          zero_division=0,
                                          labels=all_tags)
        except ImportError:
            print("⚠  scikit-learn not installed — skipping F1 computation")

        # ── Per-tag breakdown ─────────────────────────────────────────────
        result.per_tag = self._per_tag_metrics(flat_gold, flat_pred)

        # ── Confusion analysis ────────────────────────────────────────────
        result.top_confusions = self._top_confusions(flat_gold, flat_pred,
                                                      k=top_k_conf)

        # ── Bootstrap CI ──────────────────────────────────────────────────
        if bootstrap_n > 0:
            ci_low, ci_high = self._bootstrap_ci(bootstrap_n, seed)
            result.acc_ci_low  = ci_low
            result.acc_ci_high = ci_high

        # ── Dialect breakdown ─────────────────────────────────────────────
        if self.dialects:
            result.dialect_accuracy = self._dialect_breakdown()

        # ── Morphological category breakdown ──────────────────────────────
        result.morph_accuracy = self._morph_category_breakdown(
            flat_gold, flat_pred
        )

        # ── Error analysis ────────────────────────────────────────────────
        result.errors = self._error_analysis(flat_gold, flat_pred, flat_tok)

        return result

    # ── Flatten helpers ───────────────────────────────────────────────────
    def _flatten(self) -> tuple[list[str], list[str], list[str]]:
        """Flatten sentence lists, aligning gold/pred/tokens."""
        flat_gold, flat_pred, flat_tok = [], [], []
        for i, (gold_seq, pred_seq) in enumerate(
            zip(self.labels, self.predictions)
        ):
            min_len = min(len(gold_seq), len(pred_seq))
            toks    = self.tokens[i] if i < len(self.tokens) else []
            flat_gold.extend(gold_seq[:min_len])
            flat_pred.extend(pred_seq[:min_len])
            for j in range(min_len):
                flat_tok.append(toks[j] if j < len(toks) else "")
        return flat_gold, flat_pred, flat_tok

    # ── Per-tag metrics ───────────────────────────────────────────────────
    def _per_tag_metrics(
        self,
        flat_gold: list[str],
        flat_pred: list[str],
    ) -> dict:
        """Accuracy, precision, recall, F1 per POS tag."""
        tags = sorted(set(flat_gold + flat_pred))
        per_tag: dict[str, dict] = {}

        try:
            from sklearn.metrics import precision_recall_fscore_support
            p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
                flat_gold, flat_pred, labels=tags,
                average=None, zero_division=0,
            )
            for i, tag in enumerate(tags):
                correct = sum(
                    g == p == tag for g, p in zip(flat_gold, flat_pred)
                )
                support = s_arr[i]
                per_tag[tag] = {
                    "accuracy":  correct / support if support > 0 else 0.0,
                    "precision": float(p_arr[i]),
                    "recall":    float(r_arr[i]),
                    "f1":        float(f_arr[i]),
                    "support":   int(support),
                }
        except ImportError:
            # Manual fallback
            gold_c = Counter(flat_gold)
            pred_c = Counter(flat_pred)
            correct_c = Counter(
                g for g, p in zip(flat_gold, flat_pred) if g == p
            )
            for tag in tags:
                sup = gold_c[tag]
                cor = correct_c[tag]
                per_tag[tag] = {
                    "accuracy":  cor / sup if sup > 0 else 0.0,
                    "precision": 0.0,
                    "recall":    0.0,
                    "f1":        0.0,
                    "support":   sup,
                }

        return per_tag

    # ── Confusion pairs ───────────────────────────────────────────────────
    def _top_confusions(
        self,
        flat_gold: list[str],
        flat_pred: list[str],
        k: int = 20,
    ) -> list[tuple[str, str, int]]:
        """Return top-k (gold, pred) confusion pairs (excluding correct)."""
        confusion: Counter = Counter()
        for g, p in zip(flat_gold, flat_pred):
            if g != p:
                confusion[(g, p)] += 1
        return [
            (g, p, count)
            for (g, p), count in confusion.most_common(k)
        ]

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    def _bootstrap_ci(
        self, n: int = 1000, seed: int = 42, alpha: float = 0.05
    ) -> tuple[float, float]:
        """95% bootstrap CI on token accuracy."""
        rng = random.Random(seed)
        flat_gold, flat_pred, _ = self._flatten()
        pairs = list(zip(flat_gold, flat_pred))
        accs  = []
        for _ in range(n):
            sample  = rng.choices(pairs, k=len(pairs))
            correct = sum(g == p for g, p in sample)
            accs.append(correct / len(sample))
        accs.sort()
        lo = int(alpha / 2 * n)
        hi = int((1 - alpha / 2) * n)
        return accs[lo], accs[hi]

    # ── Dialect breakdown ─────────────────────────────────────────────────
    def _dialect_breakdown(self) -> dict[str, float]:
        """
        Compute accuracy per dialect.
        Requires self.dialects to be set (one label per sentence).
        """
        dialect_map = {
            "egy": "Egyptian",
            "glf": "Gulf",
            "lev": "Levantine",
            "mgr": "Maghrebi",
        }
        per_dialect: dict[str, list] = defaultdict(list)

        for i, (gold_seq, pred_seq) in enumerate(
            zip(self.labels, self.predictions)
        ):
            if i >= len(self.dialects):
                break
            dialect = self.dialects[i]
            min_len = min(len(gold_seq), len(pred_seq))
            correct = sum(
                g == p for g, p in zip(gold_seq[:min_len], pred_seq[:min_len])
            )
            per_dialect[dialect].append((correct, min_len))

        result = {}
        for d, counts in per_dialect.items():
            total_correct = sum(c for c, _ in counts)
            total_tokens  = sum(t for _, t in counts)
            acc = total_correct / total_tokens if total_tokens > 0 else 0.0
            label = dialect_map.get(d, d)
            result[label] = acc

        return result

    # ── Morphological category breakdown ──────────────────────────────────
    def _morph_category_breakdown(
        self,
        flat_gold: list[str],
        flat_pred: list[str],
    ) -> dict[str, float]:
        """
        Compute accuracy per morphological category.
        This is the 'error analysis by Arabic morphological category'
        innovative feature from the project blueprint.
        """
        category_stats: dict[str, dict[str, int]] = {
            cat: {"correct": 0, "total": 0}
            for cat in self.MORPH_CATEGORIES
        }

        for g, p in zip(flat_gold, flat_pred):
            for cat, tags in self.MORPH_CATEGORIES.items():
                if g in tags:
                    category_stats[cat]["total"] += 1
                    if g == p:
                        category_stats[cat]["correct"] += 1

        result = {}
        for cat, stats in category_stats.items():
            total = stats["total"]
            if total > 0:
                result[cat] = stats["correct"] / total

        return result

    # ── Error analysis ────────────────────────────────────────────────────
    def _error_analysis(
        self,
        flat_gold: list[str],
        flat_pred: list[str],
        flat_tok:  list[str],
        max_errors: int = 200,
    ) -> list[dict]:
        """Collect sample incorrect predictions for qualitative analysis."""
        errors = []
        for tok, g, p in zip(flat_tok, flat_gold, flat_pred):
            if g != p:
                errors.append({
                    "token": tok,
                    "gold":  g,
                    "pred":  p,
                    "gold_category": self._get_category(g),
                    "pred_category": self._get_category(p),
                })
                if len(errors) >= max_errors:
                    break
        return errors

    def _get_category(self, tag: str) -> str:
        for cat, tags in self.MORPH_CATEGORIES.items():
            if tag in tags:
                return cat
        return "other"


# ══════════════════════════════════════════════════════════════════════════════
# Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════

def pos_confusion_matrix(
    labels:      list[list[str]],
    predictions: list[list[str]],
    top_n_tags:  int = 20,
    normalize:   bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Token-level confusion matrix restricted to top-N most frequent tags.

    Args:
        labels:      Gold POS sequences.
        predictions: Predicted POS sequences.
        top_n_tags:  Only show the N most frequent tags.
        normalize:   Row-normalize the matrix.

    Returns:
        (matrix, tag_list)
    """
    flat_gold = [t for seq in labels      for t in seq]
    flat_pred = [t for seq in predictions for t in seq]

    # Restrict to top-N tags by frequency
    top_tags = [tag for tag, _ in Counter(flat_gold).most_common(top_n_tags)]

    filtered_gold = [g for g, p in zip(flat_gold, flat_pred) if g in top_tags]
    filtered_pred = [p if p in top_tags else "OTHER"
                     for g, p in zip(flat_gold, flat_pred) if g in top_tags]

    all_tags = sorted(set(filtered_gold + filtered_pred))

    try:
        from sklearn.metrics import confusion_matrix
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(all_tags)
        cm = confusion_matrix(
            le.transform(filtered_gold),
            le.transform(filtered_pred),
        )
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(
                cm.astype(float), row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums != 0,
            )
        return cm, all_tags
    except ImportError:
        return np.array([[]]), all_tags


def plot_pos_confusion_matrix(
    labels:      list[list[str]],
    predictions: list[list[str]],
    output_path: str | Path = "pos_confusion_matrix.png",
    top_n_tags:  int = 20,
    figsize:     tuple = (14, 12),
) -> None:
    """Save POS confusion matrix heatmap to disk."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠  matplotlib / seaborn not installed — skipping plot")
        return

    cm, tags = pos_confusion_matrix(labels, predictions,
                                    top_n_tags=top_n_tags, normalize=True)
    if cm.size == 0:
        print("⚠  Empty confusion matrix — skipping plot")
        return

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=tags, yticklabels=tags, ax=ax,
        linewidths=0.3, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Gold", fontsize=11)
    ax.set_title(
        f"POS Tagging Confusion Matrix (top-{top_n_tags} tags, normalized)",
        fontsize=13,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ POS confusion matrix saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Baseline comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_to_baseline(
    result:           POSResult,
    baseline_accuracy: float,
    baseline_name:    str = "Farasa POS",
) -> str:
    """
    Print a comparison between our model and a baseline (e.g. Farasa).
    Used in the jury presentation results slide.
    """
    delta = result.accuracy - baseline_accuracy
    sign  = "+" if delta >= 0 else ""
    lines = [
        "\n  POS Baseline Comparison",
        "  " + "─" * 40,
        f"  {baseline_name:<20}: {baseline_accuracy*100:.2f}%",
        f"  Our model{'':<12}: {result.accuracy*100:.2f}%",
        f"  Delta{'':<16}: {sign}{delta*100:.2f}%",
        "  " + "─" * 40,
    ]
    if result.dialect_accuracy:
        lines.append("  Dialect robustness:")
        for d, acc in sorted(result.dialect_accuracy.items()):
            d_delta = acc - baseline_accuracy
            sign_d  = "+" if d_delta >= 0 else ""
            lines.append(
                f"    {d:<12}: {acc*100:.2f}%  ({sign_d}{d_delta*100:.2f}%)"
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# W&B logging
# ══════════════════════════════════════════════════════════════════════════════

def log_to_wandb(result: POSResult, step: int | None = None) -> None:
    """Log POSResult metrics to Weights & Biases."""
    try:
        import wandb
        metrics = {
            "eval/pos_accuracy":     result.accuracy,
            "eval/pos_macro_f1":     result.macro_f1,
            "eval/pos_weighted_f1":  result.weighted_f1,
            "eval/pos_ci_low":       result.acc_ci_low,
            "eval/pos_ci_high":      result.acc_ci_high,
        }
        for tag, m in result.per_tag.items():
            safe = tag.replace("+", "_plus_")
            metrics[f"eval/pos_acc_{safe}"] = m["accuracy"]
        for dialect, acc in result.dialect_accuracy.items():
            metrics[f"eval/pos_acc_{dialect}"] = acc
        for cat, acc in result.morph_accuracy.items():
            metrics[f"eval/pos_morph_{cat}"] = acc
        wandb.log(metrics, step=step)
        print("✓ POS metrics logged to W&B")
    except ImportError:
        print("⚠  wandb not available — skipping W&B logging")


# ══════════════════════════════════════════════════════════════════════════════
# Save report
# ══════════════════════════════════════════════════════════════════════════════

def save_pos_report(
    result: POSResult,
    path: str | Path = "pos_eval_report.json",
) -> None:
    """Save full POS evaluation report to JSON."""
    report = {
        "accuracy":        result.accuracy,
        "macro_f1":        result.macro_f1,
        "weighted_f1":     result.weighted_f1,
        "acc_ci_low":      result.acc_ci_low,
        "acc_ci_high":     result.acc_ci_high,
        "n_correct":       result.n_correct,
        "n_incorrect":     result.n_incorrect,
        "n_total":         result.n_total,
        "per_tag":         result.per_tag,
        "dialect_accuracy":result.dialect_accuracy,
        "morph_accuracy":  result.morph_accuracy,
        "top_confusions":  result.top_confusions[:20],
        "errors":          result.errors[:100],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✓ POS eval report saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== POS Evaluation Sanity Check ===\n")

    # Toy data — 4 sentences with dialect labels
    gold = [
        ["NOUN", "V",    "PREP", "NOUN",  "PUNC"],
        ["PART", "V",    "NOUN", "PREP",  "PRON"],
        ["ADJ",  "NOUN", "V",    "NOUN+PRON"],
        ["V",    "PREP", "DET+NOUN", "PUNC"],
    ]
    pred = [
        ["NOUN", "V",    "PREP", "ADJ",   "PUNC"],   # 1 error: NOUN→ADJ
        ["PART", "NOUN", "NOUN", "PREP",  "PRON"],   # 1 error: V→NOUN
        ["ADJ",  "NOUN", "V",    "NOUN"],             # 1 error: NOUN+PRON→NOUN
        ["V",    "PREP", "DET+NOUN", "PUNC"],         # perfect
    ]
    tokens = [
        ["الكتاب", "يقرأ", "في", "المكتبة", "."],
        ["لما", "تحب", "حد", "من", "قلبك"],
        ["جديد", "الطالب", "يذهب", "بيته"],
        ["يسكن", "في", "القاهرة", "."],
    ]
    dialects = ["mgr", "egy", "lev", "glf"]

    evaluator = POSEvaluator(gold, pred, tokens, dialects)
    result    = evaluator.evaluate(bootstrap_n=200, seed=42)

    print(result.summary())

    # Assertions
    assert 0.0 < result.accuracy < 1.0,  "Accuracy should be partial"
    assert len(result.per_tag) > 0,      "Per-tag should be populated"
    assert len(result.top_confusions) > 0, "Should have confusions"
    assert result.acc_ci_low <= result.accuracy <= result.acc_ci_high
    assert len(result.dialect_accuracy) == 4, "All 4 dialects"
    assert len(result.morph_accuracy) > 0, "Morph breakdown populated"

    print(compare_to_baseline(result, baseline_accuracy=0.92,
                              baseline_name="Farasa POS"))

    print(f"\n  Morphological category accuracy:")
    for cat, acc in sorted(result.morph_accuracy.items()):
        print(f"    {cat:<12}: {acc*100:.1f}%")

    print(f"\n  Sample errors:")
    for e in result.errors[:5]:
        print(f"    token={e['token']:<15} gold={e['gold']:<20}"
              f" pred={e['pred']:<20}"
              f" ({e['gold_category']} → {e['pred_category']})")

    print("\n✅ All checks passed — src/evaluation/pos_eval.py ready")
