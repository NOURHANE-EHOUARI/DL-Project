"""
logging.py — Experiment tracking utilities (W&B + MLflow).

Owner: Student A
Phase: 2 — Preprocessing & Infrastructure (Week 2–3)

Usage:
    from src.utils.logging import ExperimentTracker

    tracker = ExperimentTracker(project="arabic-nlp-mtl", run_name="arabert-v2-mtl")
    tracker.log({"ner_f1": 0.87, "loss": 0.23}, step=100)
    tracker.finish()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified wrapper for Weights & Biases and MLflow tracking.

    Initialises both backends simultaneously. Gracefully degrades if
    either library is unavailable or credentials are missing.

    Args:
        project:    W&B project name.
        run_name:   Human-readable name for this run.
        config:     Hyperparameter dict logged at run start.
        use_wandb:  Enable W&B logging (default True).
        use_mlflow: Enable MLflow logging (default True).
        tags:       Optional list of tags for W&B.
    """

    def __init__(
        self,
        project: str = "arabic-nlp-mtl",
        run_name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        use_wandb: bool = True,
        use_mlflow: bool = True,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        self._wandb_run = None
        self._mlflow_run = None

        if self.use_wandb:
            self._init_wandb(project, run_name, config, tags)

        if self.use_mlflow:
            self._init_mlflow(project, run_name, config)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_wandb(
        self,
        project: str,
        run_name: Optional[str],
        config: Optional[dict],
        tags: Optional[list[str]],
    ) -> None:
        try:
            import wandb  # noqa: PLC0415

            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags,
                reinit=True,
            )
            logger.info("W&B run initialised: %s", self._wandb_run.url)
        except Exception as exc:
            logger.warning("W&B initialisation failed: %s", exc)
            self.use_wandb = False

    def _init_mlflow(
        self,
        project: str,
        run_name: Optional[str],
        config: Optional[dict],
    ) -> None:
        try:
            import mlflow  # noqa: PLC0415

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(project)
            self._mlflow_run = mlflow.start_run(run_name=run_name)

            if config:
                mlflow.log_params(config)
            logger.info("MLflow run started: %s", self._mlflow_run.info.run_id)
        except Exception as exc:
            logger.warning("MLflow initialisation failed: %s", exc)
            self.use_mlflow = False

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of scalar metrics."""
        if self.use_wandb and self._wandb_run:
            try:
                import wandb  # noqa: PLC0415
                wandb.log(metrics, step=step)
            except Exception as exc:
                logger.debug("W&B log failed: %s", exc)

        if self.use_mlflow and self._mlflow_run:
            try:
                import mlflow  # noqa: PLC0415
                mlflow.log_metrics(metrics, step=step)
            except Exception as exc:
                logger.debug("MLflow log failed: %s", exc)

    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Log a file artifact (e.g., a checkpoint)."""
        if self.use_wandb and self._wandb_run:
            try:
                import wandb  # noqa: PLC0415
                artifact = wandb.Artifact(name=artifact_type, type=artifact_type)
                artifact.add_file(artifact_path)
                self._wandb_run.log_artifact(artifact)
            except Exception as exc:
                logger.debug("W&B artifact log failed: %s", exc)

        if self.use_mlflow and self._mlflow_run:
            try:
                import mlflow  # noqa: PLC0415
                mlflow.log_artifact(artifact_path)
            except Exception as exc:
                logger.debug("MLflow artifact log failed: %s", exc)

    def finish(self) -> None:
        """Close both tracking runs cleanly."""
        if self.use_wandb and self._wandb_run:
            try:
                import wandb  # noqa: PLC0415
                wandb.finish()
            except Exception:
                pass

        if self.use_mlflow and self._mlflow_run:
            try:
                import mlflow  # noqa: PLC0415
                mlflow.end_run()
            except Exception:
                pass