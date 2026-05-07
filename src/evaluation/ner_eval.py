"""
ner_eval.py — NER Evaluation: Entity-level F1, Per-class breakdown,
              Confusion matrix, Error analysis.

Owner:   Student A
Phase:   4 — Evaluation & Analysis (Week 5–6)

Metrics computed:
    - Entity-level Precision, Recall, F1 (seqeval — strict span matching)
    - Per entity-type F1 breakdown
    - Token-level confusion matrix
    - Error analysis: FP / FN / boundary errors / type errors
    - Confidence intervals via bootstrap resampling

Dependencies:
    pip install seqeval scikit-learn pandas matplotlib seaborn
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NERResult:
    """Holds all evaluation outputs for one experiment run."""
    # Overall
    precision: float = 0.0
    recall:    float = 0.0
    f1:        float = 0.0

    # Per entity type  {type: {"precision": float, "recall": float, "f1": float, "support": int}}
    per_type: dict = field(default_factory=dict)

    # Bootstrap confidence interval on F1
    f1_ci_low:  float = 0.0
    f1_ci_high: float = 0.0

    # Error counts
    n_correct:        int = 0
    n_false_positive: int = 0
    n_false_negative: int = 0
    n_boundary_error: int = 0
    n_type_error:     int = 0

    # Raw error examples  list[dict]
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "  NER EVALUATION SUMMARY",
            "=" * 56,
            f"  Precision : {self.precision * 100:.2f}%",
            f"  Recall    : {self.recall    * 100:.2f}%",
            f"  F1        : {self.f1        * 100:.2f}%"
            f"  [{self.f1_ci_low*100:.2f} – {self.f1_ci_high*100:.2f}]",
            "",
            f"  Correct spans      : {self.n_correct}",
            f"  False positives    : {self.n_false_positive}",
            f"  False negatives    : {self.n_false_negative}",
            f"  Boundary errors    : {self.n_boundary_error}",
            f"  Type errors        : {self.n_type_error}",
            "",
            "  Per-type breakdown:",
        ]
        for etype, m in sorted(self.per_type.items(), key=lambda x: -x[1]["f1"]):
            lines.append(
                f"    {etype:<12}  P={m['precision']*100:.1f}%"
                f"  R={m['recall']*100:.1f}%"
                f"  F1={m['f1']*100:.1f}%"
                f"  (n={m['support']})"
            )
        lines.append("=" * 56)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class NEREvaluator:
    """
    Evaluates NER predictions against gold labels.

    All inputs are lists of sentences, where each sentence is a list of
    IOB2 tag strings — same format as seqeval.

    Args:
        labels:       Gold IOB2 sequences.
        predictions:  Predicted IOB2 sequences.
        tokens:       Optional word strings for error analysis readability.
    """

    def __init__(
        self,
        labels:      list[list[str]],
        predictions: list[list[str]],
        tokens:      list[list[str]] | None = None,
    ) -> None:
        assert len(labels) == len(predictions), \
            f"Length mismatch: {len(labels)} gold vs {len(predictions)} pred"
        self.labels      = labels
        self.predictions = predictions
        self.tokens      = tokens or [[] for _ in labels]

    # ── Main entry point ──────────────────────────────────────────────────
    def evaluate(
        self,
        bootstrap_n: int = 1000,
        seed: int = 42,
    ) -> NERResult:
        """
        Run full evaluation suite.

        Args:
            bootstrap_n: Number of bootstrap iterations for CI (0 = skip).
            seed:        Random seed for reproducibility.

        Returns:
            NERResult with all metrics populated.
        """
        result = NERResult()

        # ── seqeval metrics ───────────────────────────────────────────────
        try:
            from seqeval.metrics import (
                precision_score, recall_score, f1_score,
                classification_report,
            )
            result.precision = precision_score(self.labels, self.predictions)
            result.recall    = recall_score(self.labels, self.predictions)
            result.f1        = f1_score(self.labels, self.predictions)
            result.per_type  = self._per_type_metrics()
        except ImportError:
            print("⚠  seqeval not installed — run: pip install seqeval")
            result.precision, result.recall, result.f1 = \
                self._manual_prf(self.labels, self.predictions)
            result.per_type = self._per_type_metrics()

        # ── Bootstrap confidence interval ─────────────────────────────────
        if bootstrap_n > 0:
            ci_low, ci_high = self._bootstrap_ci(bootstrap_n, seed)
            result.f1_ci_low  = ci_low
            result.f1_ci_high = ci_high

        # ── Error analysis ────────────────────────────────────────────────
        counts, errors = self._error_analysis()
        result.n_correct        = counts.get("correct",  0)
        result.n_false_positive = counts.get("fp",       0)
        result.n_false_negative = counts.get("fn",       0)
        result.n_boundary_error = counts.get("boundary", 0)
        result.n_type_error     = counts.get("type",     0)
        result.errors           = errors

        return result

    # ── Per-type breakdown ────────────────────────────────────────────────
    def _per_type_metrics(self) -> dict:
        """Compute P/R/F1/support per entity type."""
        gold_spans  = self._extract_all_spans(self.labels)
        pred_spans  = self._extract_all_spans(self.predictions)

        types = set(s["type"] for s in gold_spans + pred_spans)
        per_type: dict[str, dict] = {}

        for etype in sorted(types):
            gold_e = {(s["sent"], s["start"], s["end"]) for s in gold_spans if s["type"] == etype}
            pred_e = {(s["sent"], s["start"], s["end"]) for s in pred_spans if s["type"] == etype}

            tp = len(gold_e & pred_e)
            fp = len(pred_e - gold_e)
            fn = len(gold_e - pred_e)

            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            per_type[etype] = {
                "precision": p,
                "recall":    r,
                "f1":        f1,
                "support":   len(gold_e),
            }

        return per_type

    # ── Manual P/R/F1 (fallback without seqeval) ─────────────────────────
    def _manual_prf(
        self,
        labels: list[list[str]],
        preds:  list[list[str]],
    ) -> tuple[float, float, float]:
        gold_spans = set(
            (s["sent"], s["start"], s["end"], s["type"])
            for s in self._extract_all_spans(labels)
        )
        pred_spans = set(
            (s["sent"], s["start"], s["end"], s["type"])
            for s in self._extract_all_spans(preds)
        )
        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    # ── Bootstrap CI ─────────────────────────────────────────────────────
    def _bootstrap_ci(
        self, n: int = 1000, seed: int = 42, alpha: float = 0.05
    ) -> tuple[float, float]:
        """95% bootstrap confidence interval on entity F1."""
        rng = random.Random(seed)
        indices = list(range(len(self.labels)))
        f1_scores = []

        for _ in range(n):
            sample = rng.choices(indices, k=len(indices))
            lab_s  = [self.labels[i]      for i in sample]
            pred_s = [self.predictions[i] for i in sample]
            _, _, f1 = self._manual_prf(lab_s, pred_s)
            f1_scores.append(f1)

        f1_scores.sort()
        lo = int(alpha / 2 * n)
        hi = int((1 - alpha / 2) * n)
        return f1_scores[lo], f1_scores[hi]

    # ── Error analysis ────────────────────────────────────────────────────
    def _error_analysis(self) -> tuple[dict, list[dict]]:
        """
        Classify prediction errors into:
          - correct      : exact span + type match
          - fp           : predicted span not in gold
          - fn           : gold span not predicted
          - boundary     : same type, overlapping but not exact span
          - type         : exact span, wrong type
        """
        counts = defaultdict(int)
        errors = []

        for sent_idx, (gold_tags, pred_tags, toks) in enumerate(
            zip(self.labels, self.predictions, self.tokens)
        ):
            gold_spans = {
                (s["start"], s["end"], s["type"]): s
                for s in self._sent_spans(gold_tags, sent_idx, toks)
            }
            pred_spans = {
                (s["start"], s["end"], s["type"]): s
                for s in self._sent_spans(pred_tags, sent_idx, toks)
            }

            gold_keys = set(gold_spans.keys())
            pred_keys = set(pred_spans.keys())

            # Correct
            correct = gold_keys & pred_keys
            counts["correct"] += len(correct)

            # Remaining gold / pred
            gold_rem = gold_keys - correct
            pred_rem = pred_keys - correct

            for g_key in gold_rem:
                g_start, g_end, g_type = g_key
                matched = False

                for p_key in pred_rem:
                    p_start, p_end, p_type = p_key

                    # Type error: same span, different type
                    if g_start == p_start and g_end == p_end and g_type != p_type:
                        counts["type"] += 1
                        errors.append({
                            "kind":    "type_error",
                            "sent":    sent_idx,
                            "tokens":  toks[g_start: g_end + 1],
                            "gold":    g_type,
                            "pred":    p_type,
                        })
                        matched = True
                        break

                    # Boundary error: same type, overlapping spans
                    if g_type == p_type:
                        overlap = range(max(g_start, p_start), min(g_end, p_end) + 1)
                        if len(overlap) > 0:
                            counts["boundary"] += 1
                            errors.append({
                                "kind":       "boundary_error",
                                "sent":       sent_idx,
                                "gold_span":  (g_start, g_end),
                                "pred_span":  (p_start, p_end),
                                "type":       g_type,
                                "gold_tokens": toks[g_start: g_end + 1],
                                "pred_tokens": toks[p_start: p_end + 1],
                            })
                            matched = True
                            break

                if not matched:
                    counts["fn"] += 1
                    errors.append({
                        "kind":   "false_negative",
                        "sent":   sent_idx,
                        "tokens": toks[g_start: g_end + 1],
                        "type":   g_type,
                    })

            # FP: predicted but not in gold (and not already matched)
            matched_pred = {e["pred_span"][0] for e in errors
                            if e.get("kind") == "boundary_error" and e["sent"] == sent_idx}
            for p_key in pred_rem:
                p_start, p_end, p_type = p_key
                if p_start not in matched_pred:
                    counts["fp"] += 1
                    errors.append({
                        "kind":   "false_positive",
                        "sent":   sent_idx,
                        "tokens": toks[p_start: p_end + 1],
                        "type":   p_type,
                    })

        return dict(counts), errors

    # ── Span extraction helpers ───────────────────────────────────────────
    def _extract_all_spans(self, sequences: list[list[str]]) -> list[dict]:
        spans = []
        for i, seq in enumerate(sequences):
            toks = self.tokens[i] if self.tokens else []
            spans.extend(self._sent_spans(seq, i, toks))
        return spans

    @staticmethod
    def _sent_spans(tags: list[str], sent_idx: int, tokens: list[str]) -> list[dict]:
        spans = []
        current_type  = None
        current_start = None
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if current_type:
                    spans.append({
                        "sent": sent_idx, "start": current_start,
                        "end": i - 1, "type": current_type,
                        "text": " ".join(tokens[current_start:i]) if tokens else "",
                    })
                current_type  = tag[2:]
                current_start = i
            elif tag.startswith("I-"):
                if current_type != tag[2:]:
                    if current_type:
                        spans.append({
                            "sent": sent_idx, "start": current_start,
                            "end": i - 1, "type": current_type,
                            "text": " ".join(tokens[current_start:i]) if tokens else "",
                        })
                    current_type  = tag[2:]
                    current_start = i
            else:
                if current_type:
                    spans.append({
                        "sent": sent_idx, "start": current_start,
                        "end": i - 1, "type": current_type,
                        "text": " ".join(tokens[current_start:i]) if tokens else "",
                    })
                    current_type  = None
                    current_start = None
        if current_type:
            spans.append({
                "sent": sent_idx, "start": current_start,
                "end": len(tags) - 1, "type": current_type,
                "text": " ".join(tokens[current_start:]) if tokens else "",
            })
        return spans


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def token_confusion_matrix(
    labels:      list[list[str]],
    predictions: list[list[str]],
    normalize:   bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Token-level confusion matrix over IOB2 tags.

    Returns:
        (matrix, tag_list) — matrix[i][j] = fraction of gold_tag[i]
        predicted as pred_tag[j].
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import confusion_matrix

    flat_gold = [t for seq in labels      for t in seq]
    flat_pred = [t for seq in predictions for t in seq]

    all_tags = sorted(set(flat_gold + flat_pred))
    le = LabelEncoder().fit(all_tags)

    cm = confusion_matrix(le.transform(flat_gold), le.transform(flat_pred))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.astype(float), row_sums,
                       out=np.zeros_like(cm, dtype=float),
                       where=row_sums != 0)
    return cm, all_tags


def plot_confusion_matrix(
    labels:      list[list[str]],
    predictions: list[list[str]],
    output_path: str | Path = "ner_confusion_matrix.png",
    figsize:     tuple = (14, 12),
) -> None:
    """Save a heatmap confusion matrix to disk."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠  matplotlib / seaborn not installed — skipping plot")
        return

    cm, tags = token_confusion_matrix(labels, predictions, normalize=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=tags, yticklabels=tags, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Gold", fontsize=12)
    ax.set_title("NER Token-level Confusion Matrix (normalized)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# W&B logging helper
# ---------------------------------------------------------------------------

def log_to_wandb(result: NERResult, step: int | None = None) -> None:
    """
    Log NERResult to W&B.
    Call this after evaluate() inside your training loop.
    """
    try:
        import wandb
        metrics = {
            "eval/ner_precision": result.precision,
            "eval/ner_recall":    result.recall,
            "eval/ner_f1":        result.f1,
            "eval/ner_f1_ci_low": result.f1_ci_low,
            "eval/ner_f1_ci_high":result.f1_ci_high,
            "eval/ner_fp":        result.n_false_positive,
            "eval/ner_fn":        result.n_false_negative,
            "eval/ner_boundary_errors": result.n_boundary_error,
            "eval/ner_type_errors":     result.n_type_error,
        }
        for etype, m in result.per_type.items():
            metrics[f"eval/ner_f1_{etype}"] = m["f1"]

        wandb.log(metrics, step=step)
        print("✓ NER metrics logged to W&B")
    except ImportError:
        print("⚠  wandb not available — skipping W&B logging")


# ---------------------------------------------------------------------------
# Save / load error report
# ---------------------------------------------------------------------------

def save_error_report(result: NERResult, path: str | Path = "ner_errors.json") -> None:
    """Save full error list to JSON for offline analysis."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.errors[:200], f, ensure_ascii=False, indent=2)
    print(f"✓ Error report (top 200) saved to {path}")


# ---------------------------------------------------------------------------
# Sanity check  (run: python ner_eval.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== NER Evaluation Sanity Check ===\n")

    # Toy data — 3 sentences
    gold = [
        ["O", "B-PER", "I-PER", "O", "B-LOC", "O"],
        ["B-ORG", "I-ORG", "O", "O"],
        ["O", "B-DATE", "I-DATE", "O", "B-PER", "O"],
    ]
    pred = [
        ["O", "B-PER", "I-PER", "O", "B-GPE", "O"],   # type error on LOC→GPE
        ["B-ORG", "O", "O", "O"],                       # boundary error
        ["O", "B-DATE", "I-DATE", "O", "B-PER", "O"],  # perfect
    ]
    tokens = [
        ["في", "محمد", "صلاح", "زار", "القاهرة", "اليوم"],
        ["شركة", "أرامكو", "أعلنت", "النتائج"],
        ["في", "يناير", "2024", "وصل", "المسؤول", "البارز"],
    ]

    evaluator = NEREvaluator(gold, pred, tokens)
    result = evaluator.evaluate(bootstrap_n=200, seed=42)

    print(result.summary())

    assert result.f1 > 0.0,       "F1 should be > 0"
    assert result.n_type_error >= 1, "Should detect type error"
    assert result.f1_ci_low <= result.f1 <= result.f1_ci_high, "CI should bracket F1"

    print(f"\n✓ {len(result.errors)} errors classified")
    for e in result.errors:
        print(f"    [{e['kind']}]  tokens={e.get('tokens', '?')}  "
              f"gold={e.get('gold', e.get('type', '?'))}  "
              f"pred={e.get('pred', '—')}")

    print("\n✅ All checks passed — place in src/evaluation/ner_eval.py")