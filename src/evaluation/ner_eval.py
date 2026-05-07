"""
ner_eval.py — NER Evaluation: entity-level F1, per-class breakdown, confusion matrix.

Owner: Student A
Phase: 4 — Evaluation & Analysis (Week 5–6)

Metrics computed:
  - Overall entity-level F1, Precision, Recall (seqeval)
  - Per entity-type breakdown (PER, ORG, LOC, MISC)
  - Token-level confusion matrix
  - Error analysis: false positives, false negatives, boundary errors
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def align_predictions(
    predictions: list[list[int]],
    label_ids: list[list[int]],
    id2label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert integer label IDs back to string tags, removing IGNORE_INDEX positions.

    Args:
        predictions: Model output — list of per-sentence tag ID sequences.
        label_ids:   Gold labels — list of per-sentence label ID sequences.
        id2label:    Mapping from integer ID to IOB2 tag string.

    Returns:
        (pred_tags, true_tags): aligned string sequences.
    """
    pred_tags: list[list[str]] = []
    true_tags: list[list[str]] = []

    for preds, labels in zip(predictions, label_ids):
        sentence_preds = []
        sentence_trues = []
        for pred_id, label_id in zip(preds, labels):
            if label_id == IGNORE_INDEX:
                continue  # skip subword continuations and special tokens
            sentence_preds.append(id2label.get(pred_id, "O"))
            sentence_trues.append(id2label.get(label_id, "O"))
        pred_tags.append(sentence_preds)
        true_tags.append(sentence_trues)

    return pred_tags, true_tags


def compute_ner_metrics(
    predictions: list[list[int]],
    label_ids: list[list[int]],
    id2label: dict[int, str],
    verbose: bool = True,
) -> dict[str, float]:
    """Compute full NER evaluation metrics.

    Args:
        predictions: List of per-sentence predicted tag ID sequences.
        label_ids:   List of per-sentence gold label ID sequences.
        id2label:    Integer-to-tag mapping.
        verbose:     If True, prints the seqeval classification report.

    Returns:
        Dictionary with keys:
          overall_f1, overall_precision, overall_recall,
          per_type_f1_PER, per_type_f1_ORG, per_type_f1_LOC, per_type_f1_MISC
    """
    pred_tags, true_tags = align_predictions(predictions, label_ids, id2label)

    overall_f1 = f1_score(true_tags, pred_tags, average="micro", scheme=IOB2)
    overall_precision = precision_score(true_tags, pred_tags, average="micro", scheme=IOB2)
    overall_recall = recall_score(true_tags, pred_tags, average="micro", scheme=IOB2)

    if verbose:
        report = classification_report(true_tags, pred_tags, scheme=IOB2)
        logger.info("\nNER Classification Report:\n%s", report)
        print(report)

    # Per-type F1
    per_type_report = classification_report(
        true_tags, pred_tags, scheme=IOB2, output_dict=True
    )

    metrics: dict[str, float] = {
        "ner/overall_f1": overall_f1,
        "ner/overall_precision": overall_precision,
        "ner/overall_recall": overall_recall,
    }

    for entity_type in ["PER", "ORG", "LOC", "MISC"]:
        if entity_type in per_type_report:
            metrics[f"ner/f1_{entity_type}"] = per_type_report[entity_type]["f1-score"]
            metrics[f"ner/precision_{entity_type}"] = per_type_report[entity_type]["precision"]
            metrics[f"ner/recall_{entity_type}"] = per_type_report[entity_type]["recall"]

    return metrics


def error_analysis(
    predictions: list[list[int]],
    label_ids: list[list[int]],
    tokens: list[list[str]],
    id2label: dict[int, str],
) -> dict[str, list[dict]]:
    """Categorise model errors into boundary errors, type errors, and misses.

    Args:
        predictions: Predicted tag ID sequences.
        label_ids:   Gold label ID sequences.
        tokens:      Original word tokens (aligned with labels).
        id2label:    Integer-to-tag mapping.

    Returns:
        Dict with keys: "boundary_errors", "type_errors", "false_negatives", "false_positives"
    """
    pred_tags, true_tags = align_predictions(predictions, label_ids, id2label)

    errors: dict[str, list[dict]] = defaultdict(list)

    for sent_idx, (preds, trues, sent_tokens) in enumerate(
        zip(pred_tags, true_tags, tokens)
    ):
        # Extract spans from both
        pred_spans = _extract_spans(preds)
        true_spans = _extract_spans(trues)

        # False Negatives: gold entities not predicted
        for span, etype in true_spans.items():
            if span not in pred_spans:
                errors["false_negatives"].append({
                    "sentence_idx": sent_idx,
                    "span": span,
                    "entity_type": etype,
                    "tokens": sent_tokens[span[0]:span[1] + 1],
                })

        # False Positives: predicted entities not in gold
        for span, etype in pred_spans.items():
            if span not in true_spans:
                errors["false_positives"].append({
                    "sentence_idx": sent_idx,
                    "span": span,
                    "entity_type": etype,
                    "tokens": sent_tokens[span[0]:span[1] + 1],
                })

        # Type errors: correct span, wrong entity type
        for span, pred_type in pred_spans.items():
            if span in true_spans and true_spans[span] != pred_type:
                errors["type_errors"].append({
                    "sentence_idx": sent_idx,
                    "span": span,
                    "predicted_type": pred_type,
                    "true_type": true_spans[span],
                    "tokens": sent_tokens[span[0]:span[1] + 1],
                })

    for k, v in errors.items():
        logger.info("Error category '%s': %d instances", k, len(v))

    return dict(errors)


def _extract_spans(tags: list[str]) -> dict[tuple[int, int], str]:
    """Extract entity spans from an IOB2 sequence.

    Returns:
        Dict mapping (start, end) tuples to entity type strings.
    """
    spans: dict[tuple[int, int], str] = {}
    start: Optional[int] = None
    current_type: Optional[str] = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans[(start, i - 1)] = current_type  # type: ignore[assignment]
            start = i
            current_type = tag[2:]
        elif tag.startswith("I-"):
            if start is None:
                # Invalid I without B — treat as B
                start = i
                current_type = tag[2:]
        else:  # O
            if start is not None:
                spans[(start, i - 1)] = current_type  # type: ignore[assignment]
                start = None
                current_type = None

    if start is not None:
        spans[(start, len(tags) - 1)] = current_type  # type: ignore[assignment]

    return spans