"""
src/evaluation/ablation.py — Ablation Study Runner
Owner:   Both (Student A + Student B)
Phase:   4 — Evaluation & Analysis (Week 5–6)

Runs all ablation conditions defined in the blueprint and produces
a comparison table with delta-F1 per condition.

Ablation conditions:
  1. MTL vs. 3 single-task models
  2. AraBERT v2 vs. CAMeLBERT vs. XLM-R backbone
  3. With vs. without morphological features
  4. Uncertainty loss weighting vs. fixed weights
  5. 100% vs. 50% vs. 25% training data

Usage:
    python3 -m src.evaluation.ablation --config experiments/configs/base.yaml
    python3 -m src.evaluation.ablation --dry-run   # prints plan without training
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "data" / "analysis" / "ablation_results.json"


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskScore:
    """Metrics for one task in one ablation condition."""
    ner_f1:       Optional[float] = None   # seqeval entity F1
    pos_accuracy: Optional[float] = None   # token accuracy
    coref_f1:     Optional[float] = None   # CoNLL avg F1

    def average(self) -> float:
        """Average across available task scores."""
        scores = [s for s in [self.ner_f1, self.pos_accuracy, self.coref_f1]
                  if s is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def delta(self, reference: "TaskScore") -> "TaskScore":
        """Compute delta vs. a reference condition."""
        def d(a, b):
            if a is None or b is None:
                return None
            return round(a - b, 2)
        return TaskScore(
            ner_f1=d(self.ner_f1, reference.ner_f1),
            pos_accuracy=d(self.pos_accuracy, reference.pos_accuracy),
            coref_f1=d(self.coref_f1, reference.coref_f1),
        )


@dataclass
class AblationCondition:
    """
    One ablation condition — a named experiment with a specific config change.
    """
    name:        str
    description: str
    config:      dict                    # overrides on top of base config
    scores:      Optional[TaskScore] = None
    runtime_s:   float = 0.0
    status:      str   = "pending"       # pending | running | done | failed
    error:       Optional[str] = None

    def to_row(self, reference: Optional[TaskScore] = None) -> dict:
        """Format as a table row for printing."""
        s = self.scores
        if s is None:
            return {"condition": self.name, "ner": "—", "pos": "—",
                    "coref": "—", "avg": "—", "delta_avg": "—", "status": self.status}
        delta = s.delta(reference) if reference else None
        return {
            "condition":  self.name,
            "ner":        f"{s.ner_f1:.1f}%"       if s.ner_f1       is not None else "—",
            "pos":        f"{s.pos_accuracy:.1f}%"  if s.pos_accuracy is not None else "—",
            "coref":      f"{s.coref_f1:.1f}%"      if s.coref_f1     is not None else "—",
            "avg":        f"{s.average():.2f}%",
            "delta_avg":  (f"{delta.average():+.2f}%" if delta else "REF"),
            "status":     self.status,
        }


@dataclass
class AblationReport:
    """Complete ablation study report."""
    conditions:   list[AblationCondition] = field(default_factory=list)
    reference_name: str = "Full MTL model"
    timestamp:    str   = ""
    total_runtime_s: float = 0.0

    @property
    def reference(self) -> Optional[AblationCondition]:
        for c in self.conditions:
            if c.name == self.reference_name:
                return c
        return None

    def summary_table(self) -> str:
        """Print a formatted summary table."""
        ref_scores = self.reference.scores if self.reference else None
        rows = [c.to_row(ref_scores) for c in self.conditions]

        col_w = {
            "condition": max(len(r["condition"]) for r in rows) + 2,
            "ner": 10, "pos": 10, "coref": 10, "avg": 10, "delta_avg": 12,
        }

        def fmt_row(row: dict) -> str:
            return (
                f"  {row['condition']:<{col_w['condition']}}"
                f"  {row['ner']:>{col_w['ner']}}"
                f"  {row['pos']:>{col_w['pos']}}"
                f"  {row['coref']:>{col_w['coref']}}"
                f"  {row['avg']:>{col_w['avg']}}"
                f"  {row['delta_avg']:>{col_w['delta_avg']}}"
            )

        header = fmt_row({
            "condition": "Condition", "ner": "NER F1",
            "pos": "POS Acc", "coref": "CoNLL F1",
            "avg": "Avg", "delta_avg": "Δ Avg",
        })
        sep = "  " + "─" * (len(header) - 2)

        lines = [
            "═" * len(header),
            "  ABLATION STUDY RESULTS",
            "═" * len(header),
            header, sep,
        ]
        for row in rows:
            lines.append(fmt_row(row))
        lines += [
            sep,
            f"  Reference: {self.reference_name}",
            f"  Runtime  : {self.total_runtime_s:.1f}s",
            "═" * len(header),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "reference": self.reference_name,
            "timestamp": self.timestamp,
            "total_runtime_s": self.total_runtime_s,
            "conditions": [
                {
                    "name":        c.name,
                    "description": c.description,
                    "config":      c.config,
                    "scores":      asdict(c.scores) if c.scores else None,
                    "runtime_s":   c.runtime_s,
                    "status":      c.status,
                    "error":       c.error,
                }
                for c in self.conditions
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# Ablation condition definitions
# ══════════════════════════════════════════════════════════════════════════════

def build_conditions() -> list[AblationCondition]:
    """
    Define all ablation conditions from the blueprint.
    Each condition is a named override on top of the base config.
    """
    return [
        # ── Reference ────────────────────────────────────────────────────
        AblationCondition(
            name="Full MTL model",
            description="Complete MTL system: AraBERT v2 + all 3 heads + uncertainty loss + morphology",
            config={
                "backbone":      "aubmindlab/bert-base-arabertv02",
                "tasks":         ["ner", "pos", "coref"],
                "loss_weighting":"uncertainty",
                "use_morphology": True,
                "data_fraction":  1.0,
            },
        ),

        # ── Condition 1: MTL vs. single-task ─────────────────────────────
        AblationCondition(
            name="Single-task NER only",
            description="Train NER head alone — no shared backbone signal from POS/Coref",
            config={
                "backbone": "aubmindlab/bert-base-arabertv02",
                "tasks":    ["ner"],
                "loss_weighting": "fixed",
                "use_morphology": False,
                "data_fraction":  1.0,
            },
        ),
        AblationCondition(
            name="Single-task POS only",
            description="Train POS head alone — no shared backbone signal from NER/Coref",
            config={
                "backbone": "aubmindlab/bert-base-arabertv02",
                "tasks":    ["pos"],
                "loss_weighting": "fixed",
                "use_morphology": False,
                "data_fraction":  1.0,
            },
        ),
        AblationCondition(
            name="Single-task Coref only",
            description="Train Coref head alone — no shared backbone signal from NER/POS",
            config={
                "backbone": "aubmindlab/bert-base-arabertv02",
                "tasks":    ["coref"],
                "loss_weighting": "fixed",
                "use_morphology": False,
                "data_fraction":  1.0,
            },
        ),

        # ── Condition 2: Backbone comparison ─────────────────────────────
        AblationCondition(
            name="Backbone: CAMeLBERT-Mix",
            description="Replace AraBERT v2 with CAMeLBERT-Mix — better for dialectal text",
            config={
                "backbone": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
                "tasks":    ["ner", "pos", "coref"],
                "loss_weighting": "uncertainty",
                "use_morphology": True,
                "data_fraction":  1.0,
            },
        ),
        AblationCondition(
            name="Backbone: XLM-R Large",
            description="Replace AraBERT v2 with XLM-R Large — multilingual baseline",
            config={
                "backbone": "xlm-roberta-large",
                "tasks":    ["ner", "pos", "coref"],
                "loss_weighting": "uncertainty",
                "use_morphology": True,
                "data_fraction":  1.0,
            },
        ),

        # ── Condition 3: Morphological features ──────────────────────────
        AblationCondition(
            name="Without morphology features",
            description="Remove Farasa morphological features — tests linguistic motivation",
            config={
                "backbone": "aubmindlab/bert-base-arabertv02",
                "tasks":    ["ner", "pos", "coref"],
                "loss_weighting": "uncertainty",
                "use_morphology": False,
                "data_fraction":  1.0,
            },
        ),

        # ── Condition 4: Loss weighting ───────────────────────────────────
        AblationCondition(
            name="Fixed loss weights (no uncertainty)",
            description="Replace Kendall et al. uncertainty weighting with fixed λ=1.0 per task",
            config={
                "backbone": "aubmindlab/bert-base-arabertv02",
                "tasks":    ["ner", "pos", "coref"],
                "loss_weighting": "fixed",
                "fixed_weights":  {"ner": 1.0, "pos": 1.0, "coref": 1.0},
                "use_morphology": True,
                "data_fraction":  1.0,
            },
        ),

        # ── Condition 5: Data efficiency ──────────────────────────────────
        AblationCondition(
            name="50% training data",
            description="Train on half the data — tests data efficiency",
            config={
                "backbone": "aubmindlab/bert-base-arabertv02",
                "tasks":    ["ner", "pos", "coref"],
                "loss_weighting": "uncertainty",
                "use_morphology": True,
                "data_fraction":  0.5,
            },
        ),
        AblationCondition(
            name="25% training data",
            description="Train on quarter of the data — low-resource scenario",
            config={
                "backbone": "aubmindlab/bert-base-arabertv02",
                "tasks":    ["ner", "pos", "coref"],
                "loss_weighting": "uncertainty",
                "use_morphology": True,
                "data_fraction":  0.25,
            },
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

class AblationRunner:
    """
    Runs all ablation conditions sequentially and saves results.

    In dry-run mode, prints the experiment plan without training.
    In real mode, calls the trainer for each condition.

    Args:
        conditions:  List of AblationCondition to run.
        output_path: Where to save the JSON report.
        dry_run:     If True, simulate without actual training.
    """

    def __init__(
        self,
        conditions:  list[AblationCondition],
        output_path: Path = RESULTS,
        dry_run:     bool = False,
    ):
        self.conditions  = conditions
        self.output_path = output_path
        self.dry_run     = dry_run
        self.report      = AblationReport(conditions=conditions)

    def run(self) -> AblationReport:
        """Run all conditions and return the report."""
        import datetime
        self.report.timestamp = datetime.datetime.now().isoformat()
        t_start = time.perf_counter()

        print(f"\n{'═'*60}")
        print(f"  ABLATION STUDY — {len(self.conditions)} conditions")
        print(f"  Mode: {'DRY RUN' if self.dry_run else 'TRAINING'}")
        print(f"{'═'*60}\n")

        for i, condition in enumerate(self.conditions, 1):
            print(f"[{i}/{len(self.conditions)}] {condition.name}")
            print(f"  {condition.description}")

            t0 = time.perf_counter()
            condition.status = "running"

            try:
                if self.dry_run:
                    scores = self._simulate(condition)
                else:
                    scores = self._train_and_evaluate(condition)

                condition.scores    = scores
                condition.status    = "done"
                condition.runtime_s = time.perf_counter() - t0

                row = condition.to_row(
                    self.report.reference.scores
                    if self.report.reference else None
                )
                print(f"  NER={row['ner']}  POS={row['pos']}  "
                      f"Coref={row['coref']}  Δ={row['delta_avg']}  "
                      f"({condition.runtime_s:.1f}s)")

            except Exception as e:
                condition.status = "failed"
                condition.error  = str(e)
                logger.error("Condition '%s' failed: %s", condition.name, e)
                print(f"  FAILED: {e}")

            print()
            self._save_checkpoint()

        self.report.total_runtime_s = time.perf_counter() - t_start
        print(self.report.summary_table())
        self._save_checkpoint()
        print(f"\n  Report saved → {self.output_path}")
        return self.report

    def _simulate(self, condition: AblationCondition) -> TaskScore:
        """
        Simulate scores for dry-run mode.
        Applies realistic deltas based on condition type.
        """
        import random
        rng = random.Random(hash(condition.name) % 2**32)

        # Base scores (reference MTL model)
        base_ner, base_pos, base_coref = 86.2, 96.8, 68.8

        # Apply realistic deltas per condition
        deltas = {
            "Single-task NER only":          (-2.8,   None,  None),
            "Single-task POS only":          ( None,  -1.7,  None),
            "Single-task Coref only":        ( None,   None, -7.5),
            "Backbone: CAMeLBERT-Mix":       (-1.2,  -0.5,  -1.8),
            "Backbone: XLM-R Large":         (-1.3,  -0.8,  -2.0),
            "Without morphology features":   (-1.5,  -1.2,  -2.7),
            "Fixed loss weights (no uncertainty)": (-1.1, -0.6, -1.4),
            "50% training data":             (-4.9,  -2.6,  -7.1),
            "25% training data":             (-8.7,  -4.8, -12.3),
        }

        d_ner, d_pos, d_coref = deltas.get(condition.name, (0.0, 0.0, 0.0))

        def jitter(v): return round(v + rng.uniform(-0.3, 0.3), 1)

        return TaskScore(
            ner_f1=jitter(base_ner + d_ner) if d_ner is not None else None,
            pos_accuracy=jitter(base_pos + d_pos) if d_pos is not None else None,
            coref_f1=jitter(base_coref + d_coref) if d_coref is not None else None,
        )

    def _train_and_evaluate(self, condition: AblationCondition) -> TaskScore:
        """
        Real training + evaluation for one condition.
        Calls the MTL trainer with the condition's config overrides.
        """
        # Import here to avoid circular imports
        try:
            from src.training.trainer import MTLTrainer
        except ImportError:
            raise ImportError(
                "MTLTrainer not available. "
                "Run with --dry-run to simulate without training."
            )

        cfg = condition.config.copy()
        trainer = MTLTrainer(
            backbone=cfg.get("backbone", "aubmindlab/bert-base-arabertv02"),
            tasks=cfg.get("tasks", ["ner", "pos", "coref"]),
            loss_weighting=cfg.get("loss_weighting", "uncertainty"),
            use_morphology=cfg.get("use_morphology", True),
            data_fraction=cfg.get("data_fraction", 1.0),
        )
        metrics = trainer.train_and_evaluate()
        return TaskScore(
            ner_f1=metrics.get("ner_f1"),
            pos_accuracy=metrics.get("pos_accuracy"),
            coref_f1=metrics.get("coref_f1"),
        )

    def _save_checkpoint(self):
        """Save current results to disk after each condition."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_report(path: Path = RESULTS) -> AblationReport:
    """Load a saved ablation report from disk."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    conditions = []
    for c in data["conditions"]:
        scores = None
        if c["scores"]:
            scores = TaskScore(**c["scores"])
        conditions.append(AblationCondition(
            name=c["name"],
            description=c["description"],
            config=c["config"],
            scores=scores,
            runtime_s=c["runtime_s"],
            status=c["status"],
            error=c.get("error"),
        ))

    report = AblationReport(
        conditions=conditions,
        reference_name=data["reference"],
        timestamp=data["timestamp"],
        total_runtime_s=data["total_runtime_s"],
    )
    return report


def compare_reports(r1: AblationReport, r2: AblationReport,
                    label1: str = "Run 1", label2: str = "Run 2") -> str:
    """
    Compare two ablation reports side by side.
    Useful for comparing results across backbone checkpoints.
    """
    lines = [
        f"\n{'═'*70}",
        f"  ABLATION COMPARISON: {label1} vs. {label2}",
        f"{'═'*70}",
        f"  {'Condition':<35}  {label1:>12}  {label2:>12}  {'Δ':>8}",
        f"  {'─'*35}  {'─'*12}  {'─'*12}  {'─'*8}",
    ]

    c1_map = {c.name: c for c in r1.conditions}
    c2_map = {c.name: c for c in r2.conditions}

    for name in c1_map:
        c1 = c1_map.get(name)
        c2 = c2_map.get(name)
        avg1 = f"{c1.scores.average():.2f}%" if c1 and c1.scores else "—"
        avg2 = f"{c2.scores.average():.2f}%" if c2 and c2.scores else "—"
        delta_str = "—"
        if c1 and c2 and c1.scores and c2.scores:
            delta = c2.scores.average() - c1.scores.average()
            delta_str = f"{delta:+.2f}%"
        lines.append(f"  {name:<35}  {avg1:>12}  {avg2:>12}  {delta_str:>8}")

    lines.append(f"{'═'*70}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ablation study for Arabic MTL NLP project."
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Simulate scores without real training")
    p.add_argument("--config", default="experiments/configs/base.yaml",
                   help="Base YAML config path")
    p.add_argument("--output", default=str(RESULTS),
                   help="Output JSON path for results")
    p.add_argument("--conditions", nargs="*",
                   help="Run only specific conditions by name (default: all)")
    p.add_argument("--load", action="store_true",
                   help="Load and display a saved report without running")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s — %(levelname)s — %(message)s")
    args = parse_args()

    if args.load:
        report = load_report(Path(args.output))
        print(report.summary_table())
    else:
        conditions = build_conditions()

        # Filter if specific conditions requested
        if args.conditions:
            conditions = [c for c in conditions
                          if any(name.lower() in c.name.lower()
                                 for name in args.conditions)]
            if not conditions:
                print(f"No conditions matched: {args.conditions}")
                exit(1)

        runner = AblationRunner(
            conditions=conditions,
            output_path=Path(args.output),
            dry_run=args.dry_run,
        )
        runner.run()
