"""
MLflow experiment tracking integration for BWF models.
Wraps MLflow calls to provide a clean interface for logging
parameters, metrics, artifacts, and registering models.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.config import settings
from src.utils.logger import logger


class MLflowTracker:
    """
    Manages MLflow experiment tracking for BWF ranking models.

    Usage:
        tracker = MLflowTracker(draw="MS", region="Asia")
        with tracker.start_run("lgbm_v1") as run:
            tracker.log_params({"n_estimators": 200, "lr": 0.05})
            tracker.log_metrics({"mae": 1234.5, "r2": 0.87})
            tracker.log_model(model, "lgbm_model")
    """

    def __init__(self, draw: str, region: str = "Global") -> None:
        self.draw = draw.upper()
        self.region = region
        self.experiment_name = f"{settings.MLFLOW_EXPERIMENT_PREFIX}_{self.draw}_{self.region}"
        self.client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

        # Set tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self._ensure_experiment()

    # ── Public API ───────────────────────────────────────────────────────────

    @contextmanager
    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[Run, None, None]:
        """Context manager to wrap an MLflow run."""
        default_tags = {
            "draw": self.draw,
            "region": self.region,
            "model_type": "lightgbm",
            "project": "bwf-ranking-prediction",
        }
        if tags:
            default_tags.update(tags)

        logger.info(f"Starting MLflow run: [{self.experiment_name}] {run_name}")

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=default_tags,
        ) as run:
            yield run
            logger.success(f"MLflow run complete: {run.info.run_id[:8]}...")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to the active run."""
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} params to MLflow")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics to the active run."""
        mlflow.log_metrics(metrics, step=step)
        summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.debug(f"Logged metrics: {summary}")

    def log_model(self, model: Any, artifact_path: str = "model") -> str:
        """Log the trained LightGBM model as an MLflow artifact."""
        mlflow.lightgbm.log_model(model, artifact_path=artifact_path)
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        logger.info(f"Model logged to: {model_uri}")
        return model_uri

    def log_feature_importance(self, model: Any, feature_names: list) -> None:
        """Save feature importance as a CSV artifact."""
        import pandas as pd
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        tmp_path = Path("reports") / f"feature_importance_{self.draw}_{self.region}.csv"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(tmp_path, index=False)
        mlflow.log_artifact(str(tmp_path))
        logger.debug(f"Feature importance logged: {tmp_path.name}")

    def register_model(self, model_uri: str, model_name: str, stage: str = "Staging") -> None:
        """Register model in MLflow Model Registry and transition to stage."""
        registered = mlflow.register_model(model_uri, model_name)
        self.client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage=stage,
        )
        logger.success(f"Model registered: {model_name} v{registered.version} → {stage}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _ensure_experiment(self) -> None:
        """Create the MLflow experiment if it doesn't exist."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=str(Path("mlruns") / self.experiment_name),
            )
            logger.info(f"Created MLflow experiment: {self.experiment_name} (id={experiment_id})")
        else:
            experiment_id = experiment.experiment_id
        self.experiment_id = experiment_id
