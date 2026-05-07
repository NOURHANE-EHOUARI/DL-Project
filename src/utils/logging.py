"""
logging.py — Experiment Tracking with Weights & Biases.

Owner: Student A
Phase: 2 — Preprocessing & Infrastructure (Week 2–3)

Centralizes all W&B interactions:
  - Run initialization with full config logging
  - Metric logging (loss, F1, accuracy, CoNLL scores)
  - Model artifact saving and versioning
  - Hyperparameter sweep configuration
  - Graceful no-op fallback when W&B is disabled (e.g. CI, offline)

Usage:
    from src.utils.logging import ExperimentTracker

    tracker = ExperimentTracker(config)
    tracker.log({"train/ner_loss": 0.45, "train/pos_loss": 0.12}, step=100)
    tracker.log_metrics(phase="eval", ner_f1=0.87, pos_acc=0.96, coref_f1=0.63)
    tracker.save_model_artifact("checkpoints/best_ner.pt", name="best-ner")
    tracker.finish()
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Run: pip install wandb. Tracking disabled.")


# ---------------------------------------------------------------------------
# Experiment Configuration Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Full experiment configuration logged to W&B at run start.

    Every hyperparameter that could affect results must appear here.
    This ensures every run is fully reproducible from the W&B dashboard.
    """
    # ── Identity ──────────────────────────────────────────────────────
    project: str = "arabic-nlp-project"
    run_name: str = "mtl-run"
    group: str = "mtl"           # groups related runs (e.g. "ablation-backbone")
    tags: list[str] = field(default_factory=lambda: ["mtl", "arabic", "ner", "pos", "coref"])
    notes: str = ""

    # ── Backbone ──────────────────────────────────────────────────────
    backbone: str = "arabert-v2"          # key from SUPPORTED_BACKBONES
    hidden_dropout_prob: float = 0.1
    freeze_backbone: bool = False
    gradient_checkpointing: bool = False

    # ── Training ──────────────────────────────────────────────────────
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 32
    coref_batch_size: int = 8            # coref needs smaller batch (O(n²) spans)
    max_seq_length: int = 512
    warmup_ratio: float = 0.1            # fraction of steps for LR warmup
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    fp16: bool = False

    # ── MTL Loss Weighting ────────────────────────────────────────────
    loss_weighting: str = "uncertainty"  # "uncertainty" | "fixed" | "equal"
    ner_loss_weight: float = 1.0         # used only when loss_weighting="fixed"
    pos_loss_weight: float = 1.0
    coref_loss_weight: float = 1.0

    # ── Task Sampling ─────────────────────────────────────────────────
    task_sampling: str = "uniform"       # "uniform" | "proportional" | "annealed"

    # ── NER Head ──────────────────────────────────────────────────────
    ner_lstm_hidden: int = 256
    ner_lstm_layers: int = 1
    ner_use_crf: bool = True

    # ── Coref Head ────────────────────────────────────────────────────
    coref_max_span_width: int = 30
    coref_top_span_ratio: float = 0.4
    coref_ffnn_depth: int = 2
    coref_ffnn_size: int = 1000

    # ── Morphological Features ────────────────────────────────────────
    use_morphological_features: bool = True   # inject Farasa features
    morph_feature_dim: int = 64

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = 42


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Central interface for all W&B logging in the project.

    Wraps wandb with a clean API and a graceful no-op fallback
    when W&B is unavailable or explicitly disabled.

    Args:
        config:  ExperimentConfig dataclass with full hyperparameters.
        enabled: Set to False to disable W&B (useful for quick debug runs).
                 Also reads WANDB_DISABLED=true from environment.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        enabled: bool = True,
    ) -> None:
        self.config = config
        self._enabled = enabled and WANDB_AVAILABLE
        self._enabled = self._enabled and not _is_wandb_disabled()
        self._run = None

        if self._enabled:
            self._run = wandb.init(
                project=config.project,
                name=config.run_name,
                group=config.group,
                tags=config.tags,
                notes=config.notes,
                config=asdict(config),
                reinit=True,
            )
            logger.info(
                "W&B run initialized: %s | project: %s | url: %s",
                config.run_name,
                config.project,
                wandb.run.url,
            )
        else:
            logger.info("ExperimentTracker running in no-op mode (W&B disabled).")

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log an arbitrary dict of metrics.

        Args:
            metrics: e.g. {"train/ner_loss": 0.45, "train/pos_loss": 0.12}
            step:    Global training step. If None, W&B auto-increments.
        """
        if self._enabled:
            wandb.log(metrics, step=step)

    def log_metrics(
        self,
        phase: str,                          # "train" | "eval" | "test"
        step: Optional[int] = None,
        ner_loss: Optional[float] = None,
        pos_loss: Optional[float] = None,
        coref_loss: Optional[float] = None,
        total_loss: Optional[float] = None,
        ner_f1: Optional[float] = None,
        ner_precision: Optional[float] = None,
        ner_recall: Optional[float] = None,
        pos_accuracy: Optional[float] = None,
        coref_muc: Optional[float] = None,
        coref_b3: Optional[float] = None,
        coref_ceafe: Optional[float] = None,
        coref_avg: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log structured metrics with consistent naming convention.

        All metrics are namespaced by phase: "train/ner_f1", "eval/pos_accuracy", etc.
        This creates clean, comparable charts in the W&B dashboard.
        """
        metrics: dict[str, Any] = {}

        # Loss metrics
        if ner_loss    is not None: metrics[f"{phase}/ner_loss"]    = ner_loss
        if pos_loss    is not None: metrics[f"{phase}/pos_loss"]    = pos_loss
        if coref_loss  is not None: metrics[f"{phase}/coref_loss"]  = coref_loss
        if total_loss  is not None: metrics[f"{phase}/total_loss"]  = total_loss

        # NER metrics
        if ner_f1        is not None: metrics[f"{phase}/ner_f1"]        = ner_f1
        if ner_precision is not None: metrics[f"{phase}/ner_precision"] = ner_precision
        if ner_recall    is not None: metrics[f"{phase}/ner_recall"]    = ner_recall

        # POS metrics
        if pos_accuracy is not None: metrics[f"{phase}/pos_accuracy"] = pos_accuracy

        # Coreference metrics (CoNLL standard)
        if coref_muc   is not None: metrics[f"{phase}/coref_muc"]   = coref_muc
        if coref_b3    is not None: metrics[f"{phase}/coref_b3"]    = coref_b3
        if coref_ceafe is not None: metrics[f"{phase}/coref_ceafe"] = coref_ceafe
        if coref_avg   is not None: metrics[f"{phase}/coref_avg"]   = coref_avg

        # Any extra metrics
        for key, value in extra.items():
            metrics[f"{phase}/{key}"] = value

        if metrics:
            self.log(metrics, step=step)

    def log_summary(self, key: str, value: Any) -> None:
        """Log a run-level summary metric (e.g. best eval F1).

        Summary metrics appear prominently in the W&B run table,
        making it easy to compare runs at a glance.
        """
        if self._enabled:
            wandb.run.summary[key] = value

    # ------------------------------------------------------------------
    # Artifact management
    # ------------------------------------------------------------------

    def save_model_artifact(
        self,
        checkpoint_path: str | Path,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save a model checkpoint as a W&B artifact.

        Artifacts are versioned automatically. Use this to save:
          - Best checkpoint per task ("best-ner", "best-pos", "best-coref")
          - Final MTL checkpoint ("mtl-final")

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            name:            Artifact name (e.g. "best-ner").
            artifact_type:   W&B artifact type tag.
            metadata:        Optional dict logged alongside the artifact
                             (e.g. {"ner_f1": 0.87, "epoch": 8}).
        """
        if not self._enabled:
            return

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning("Artifact not saved: path does not exist: %s", checkpoint_path)
            return

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata or {},
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
        logger.info("Artifact saved to W&B: %s (%s)", name, checkpoint_path)

    def save_dataset_artifact(
        self,
        data_dir: str | Path,
        name: str = "arabic-nlp-datasets",
    ) -> None:
        """Save processed dataset files as a W&B artifact for reproducibility."""
        if not self._enabled:
            return

        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_dir(str(data_dir))
        wandb.log_artifact(artifact)
        logger.info("Dataset artifact saved: %s", name)

    # ------------------------------------------------------------------
    # Sweep utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_sweep_config() -> dict[str, Any]:
        """Returns the W&B sweep configuration for hyperparameter search.

        Run a sweep with:
            sweep_id = wandb.sweep(
                ExperimentTracker.get_sweep_config(),
                project="arabic-nlp-project"
            )
            wandb.agent(sweep_id, function=train, count=30)

        Covers the hyperparameters most likely to affect MTL performance.
        """
        return {
            "method": "bayes",           # Bayesian optimization (better than grid/random)
            "metric": {
                "name": "eval/ner_f1",   # primary metric to maximize
                "goal": "maximize",
            },
            "parameters": {
                "learning_rate": {
                    "distribution": "log_uniform_values",
                    "min": 1e-5,
                    "max": 5e-5,
                },
                "hidden_dropout_prob": {
                    "values": [0.1, 0.2, 0.3],
                },
                "ner_lstm_hidden": {
                    "values": [128, 256, 512],
                },
                "warmup_ratio": {
                    "values": [0.06, 0.1, 0.15],
                },
                "loss_weighting": {
                    "values": ["uncertainty", "fixed", "equal"],
                },
                "task_sampling": {
                    "values": ["uniform", "proportional"],
                },
                "use_morphological_features": {
                    "values": [True, False],
                },
                "coref_top_span_ratio": {
                    "values": [0.3, 0.4, 0.5],
                },
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 3,
            },
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def watch_model(self, model: Any, log_freq: int = 100) -> None:
        """Track model gradients and parameter histograms in W&B.

        Useful for detecting vanishing/exploding gradients during training.
        Call once after model initialization, before the training loop.
        """
        if self._enabled:
            wandb.watch(model, log="all", log_freq=log_freq)
            logger.info("W&B model watch enabled (log_freq=%d)", log_freq)

    def finish(self) -> None:
        """Close the W&B run. Always call this at the end of training."""
        if self._enabled and self._run is not None:
            wandb.finish()
            logger.info("W&B run finished.")

    @property
    def run_id(self) -> Optional[str]:
        """Returns the W&B run ID, or None if tracking is disabled."""
        if self._enabled and self._run is not None:
            return wandb.run.id
        return None

    @property
    def run_url(self) -> Optional[str]:
        """Returns the W&B run URL for sharing with your project partner."""
        if self._enabled and self._run is not None:
            return wandb.run.url
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_wandb_disabled() -> bool:
    """Check if W&B is disabled via environment variable."""
    return os.environ.get("WANDB_DISABLED", "false").lower() in ("true", "1", "yes")


def setup_python_logging(level: int = logging.INFO) -> None:
    """Configure Python's root logger for clean console output.

    Call once at the entry point of training scripts.
    """
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    # Suppress overly verbose HuggingFace logs
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)