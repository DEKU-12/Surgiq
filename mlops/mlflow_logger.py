"""
SurgIQ — MLflow Experiment Logger
===================================
Logs training runs, model metrics, and pipeline session stats to MLflow.

Tracks:
  - Classifier training: loss/accuracy per epoch, best val acc, hyperparams
  - Detector training  : mAP@0.5, mAP@0.5:0.95, per-class AP
  - Pipeline sessions  : per-frame latency, instrument activity distribution

Usage:
    from mlops.mlflow_logger import SurgIQLogger

    logger = SurgIQLogger()

    # Log a classifier training run
    with logger.start_run("classifier_training"):
        logger.log_params({"epochs": 30, "lr": 1e-4, "batch_size": 32})
        for epoch, metrics in training_loop():
            logger.log_metrics(metrics, step=epoch)
        logger.log_model_summary(best_val_acc=0.947)
"""

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import config as cfg

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    raise ImportError("mlflow not installed. Run: pip install mlflow")


class SurgIQLogger:
    """
    Thin wrapper around MLflow for SurgIQ experiment tracking.

    All runs are logged to the configured MLflow tracking URI
    under the 'surgiq-sessions' experiment.
    """

    def __init__(
        self,
        tracking_uri: str = cfg.MLFLOW_TRACKING_URI,
        experiment_name: str = cfg.MLFLOW_EXPERIMENT_NAME,
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = None
        print(f"[MLflowLogger] tracking_uri={tracking_uri}")
        print(f"[MLflowLogger] experiment={experiment_name}")

    # ── Context manager ───────────────────────────────────────────────────────

    def start_run(self, run_name: str):
        """Start an MLflow run. Use as context manager."""
        self._run = mlflow.start_run(run_name=run_name)
        print(f"[MLflowLogger] Started run: {run_name}  id={self._run.info.run_id}")
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._run:
            mlflow.end_run()
            self._run = None

    # ── Logging helpers ───────────────────────────────────────────────────────

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics (optionally at a training step/epoch)."""
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Upload a file (e.g. confusion matrix PNG) as an artifact."""
        mlflow.log_artifact(local_path)

    def log_model_summary(self, best_val_acc: float) -> None:
        """Log final classifier summary metrics."""
        mlflow.log_metric("best_val_accuracy", best_val_acc)

    # ── Classifier run ────────────────────────────────────────────────────────

    def log_classifier_run(
        self,
        params: dict,
        history: list[dict],
        best_val_acc: float,
        weights_path: Path | None = None,
        confusion_matrix_path: Path | None = None,
    ) -> str:
        """
        Log a complete classifier training run to MLflow.

        Parameters
        ----------
        params               : dict  Hyperparameters (epochs, lr, batch_size, etc.)
        history              : list  Per-epoch dicts with train_loss, val_loss, etc.
        best_val_acc         : float Best validation accuracy achieved.
        weights_path         : Path  Optional path to best.pth to upload.
        confusion_matrix_path: Path  Optional confusion matrix PNG.

        Returns
        -------
        str  MLflow run ID.
        """
        with mlflow.start_run(run_name="classifier_training") as run:
            # Hyperparameters
            mlflow.log_params(params)

            # Per-epoch metrics
            for row in history:
                step = row["epoch"]
                mlflow.log_metrics({
                    "train_loss": row["train_loss"],
                    "train_acc" : row["train_acc"],
                    "val_loss"  : row["val_loss"],
                    "val_acc"   : row["val_acc"],
                }, step=step)

            # Summary
            mlflow.log_metric("best_val_accuracy", best_val_acc)

            # Artifacts
            if weights_path and Path(weights_path).exists():
                mlflow.log_artifact(str(weights_path), artifact_path="weights")

            if confusion_matrix_path and Path(confusion_matrix_path).exists():
                mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")

            run_id = run.info.run_id
            print(f"[MLflowLogger] Classifier run logged: {run_id}")
            return run_id

    # ── Detector run ──────────────────────────────────────────────────────────

    def log_detector_run(
        self,
        params: dict,
        metrics: dict,
        weights_path: Path | None = None,
    ) -> str:
        """
        Log a YOLO detector training run.

        Parameters
        ----------
        params  : dict  Training params (epochs, imgsz, batch, etc.)
        metrics : dict  Final metrics (mAP50, mAP50_95, precision, recall, etc.)
        """
        with mlflow.start_run(run_name="detector_training") as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if weights_path and Path(weights_path).exists():
                mlflow.log_artifact(str(weights_path), artifact_path="weights")

            run_id = run.info.run_id
            print(f"[MLflowLogger] Detector run logged: {run_id}")
            return run_id

    # ── Pipeline session ──────────────────────────────────────────────────────

    def log_pipeline_session(
        self,
        session_id: str,
        frame_times_ms: list[float],
        label_counts: dict[str, int],
        source: str,
    ) -> str:
        """
        Log a pipeline inference session (latency + activity breakdown).
        """
        import numpy as np

        with mlflow.start_run(run_name=f"session_{session_id}") as run:
            total = sum(label_counts.values()) or 1

            mlflow.log_params({
                "session_id": session_id,
                "source"    : str(source),
            })

            mlflow.log_metrics({
                "total_frames"      : total,
                "avg_latency_ms"    : float(np.mean(frame_times_ms)) if frame_times_ms else 0,
                "p95_latency_ms"    : float(np.percentile(frame_times_ms, 95)) if frame_times_ms else 0,
                "fps_effective"     : 1000 / np.mean(frame_times_ms) if frame_times_ms else 0,
                "pct_no_instrument" : label_counts.get("no_instrument", 0) / total * 100,
                "pct_grasper_only"  : label_counts.get("grasper_only", 0)  / total * 100,
                "pct_hook_only"     : label_counts.get("hook_only", 0)     / total * 100,
                "pct_both"          : label_counts.get("both_instruments", 0) / total * 100,
            })

            run_id = run.info.run_id
            print(f"[MLflowLogger] Session logged: {run_id}")
            return run_id
