"""
Enhanced model trainer with MLflow tracking + Optuna tuning.
Phase 2 upgrade of the original trainer.py
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.lightgbm

from src.config import settings
from src.utils.logger import logger
from src.mlops.tracker import MLflowTracker
from src.mlops.tuner import HyperparameterTuner

try:
    from lightgbm import LGBMRegressor
    import xgboost as xgb
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

SUPPORTED_MODELS = ["lightgbm", "xgboost", "gradient_boosting"]


class BWFModelTrainer:
    """
    Phase 2 trainer: trains models with MLflow tracking, Optuna tuning,
    walk-forward CV, and supports multiple algorithm comparison.

    Args:
        draw:       BWF draw code (MS/WS/MD/WD/XD).
        region:     Continent (Global/Asia/Europe).
        model_type: Algorithm to use ('lightgbm', 'xgboost', 'gradient_boosting').
        use_mlflow: Whether to track experiments in MLflow.
        tune_params: Whether to run Optuna hyperparameter search first.

    Usage:
        trainer = BWFModelTrainer(draw="MS", region="Asia", tune_params=True)
        model, metrics = trainer.train(features_df, feature_cols)
        trainer.save()
    """

    def __init__(
        self,
        draw: str,
        region: str = "Global",
        model_type: str = "lightgbm",
        use_mlflow: bool = True,
        tune_params: bool = False,
        n_tune_trials: int = 30,
    ) -> None:
        assert model_type in SUPPORTED_MODELS, f"model_type must be one of {SUPPORTED_MODELS}"
        self.draw = draw.upper()
        self.region = region
        self.model_type = model_type
        self.use_mlflow = use_mlflow
        self.tune_params = tune_params
        self.n_tune_trials = n_tune_trials

        self.model_name = f"{model_type}_{self.draw}_{self.region}".lower()
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.best_params: Optional[Dict] = None

        if use_mlflow:
            self.tracker = MLflowTracker(draw=self.draw, region=self.region)

    # ── Public API ───────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "points",
        params: Optional[Dict] = None,
        n_splits: int = 5,
    ) -> Tuple[any, Dict[str, float]]:
        """
        Full training pipeline:
          1. (Optional) Optuna hyperparameter search
          2. Walk-forward cross-validation
          3. Final model fit on full data
          4. MLflow logging

        Returns:
            (trained_model, cv_metrics_dict)
        """
        logger.info(
            f"[Phase 2] Training {self.model_type.upper()} | "
            f"draw={self.draw}, region={self.region} | "
            f"samples={len(df):,}, features={len(feature_cols)}"
        )

        X = df[feature_cols].values
        y = df[target_col].values
        self.feature_names = feature_cols

        # ── Step 1: Hyperparameter tuning ──────────────────────────────────
        if self.tune_params and self.model_type == "lightgbm":
            logger.info("Running Optuna hyperparameter search...")
            tuner = HyperparameterTuner(draw=self.draw, region=self.region, n_splits=n_splits)
            self.best_params = tuner.optimize(X, y, n_trials=self.n_tune_trials)
            study_summary = tuner.get_study_summary()
        else:
            self.best_params = params or self._default_params()
            study_summary = {}

        # ── Step 2: Walk-forward CV ─────────────────────────────────────────
        cv_metrics = self._walk_forward_cv(X, y, self.best_params, n_splits)

        # ── Step 3: Final fit on all data ──────────────────────────────────
        self.model = self._build_model(self.best_params)
        self.model.fit(X, y)

        avg_metrics = {
            "mae_cv": float(np.mean([m["mae"] for m in cv_metrics])),
            "rmse_cv": float(np.mean([m["rmse"] for m in cv_metrics])),
            "r2_cv": float(np.mean([m["r2"] for m in cv_metrics])),
            "mae_std": float(np.std([m["mae"] for m in cv_metrics])),
        }

        # ── Step 4: MLflow logging ─────────────────────────────────────────
        if self.use_mlflow:
            run_name = f"{self.model_type}_{self.draw}_{self.region}"
            with self.tracker.start_run(run_name) as run:
                self.tracker.log_params({
                    **self.best_params,
                    "draw": self.draw,
                    "region": self.region,
                    "model_type": self.model_type,
                    "train_samples": len(X),
                    "n_features": len(feature_cols),
                    "cv_splits": n_splits,
                    "tuned": self.tune_params,
                })
                self.tracker.log_metrics(avg_metrics)

                if study_summary:
                    mlflow.log_param("optuna_n_trials", study_summary.get("n_trials", 0))
                    mlflow.log_param("optuna_best_trial", study_summary.get("best_trial", 0))

                model_uri = self.tracker.log_model(self.model, artifact_path="model")

                if hasattr(self.model, "feature_importances_"):
                    self.tracker.log_feature_importance(self.model, feature_cols)

                self._mlflow_run_id = run.info.run_id

        logger.success(
            f"Training complete: MAE={avg_metrics['mae_cv']:.1f} ± {avg_metrics['mae_std']:.1f}, "
            f"R²={avg_metrics['r2_cv']:.4f}"
        )
        return self.model, avg_metrics

    def compare_models(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "points",
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """
        Train and compare LightGBM vs XGBoost vs GradientBoosting.
        Logs all three to MLflow for easy comparison.

        Returns:
            DataFrame with model comparison results.
        """
        logger.info(f"Comparing {len(SUPPORTED_MODELS)} models for {self.draw}/{self.region}...")
        results = []
        X = df[feature_cols].values
        y = df[target_col].values

        for algo in SUPPORTED_MODELS:
            logger.info(f"  Training {algo}...")
            params = self._default_params(algo)
            cv_metrics = self._walk_forward_cv(X, y, params, n_splits, model_type=algo)
            avg_mae = np.mean([m["mae"] for m in cv_metrics])
            avg_r2 = np.mean([m["r2"] for m in cv_metrics])

            if self.use_mlflow:
                with self.tracker.start_run(f"{algo}_compare", tags={"comparison": "True"}):
                    self.tracker.log_params({**params, "model_type": algo})
                    self.tracker.log_metrics({"mae_cv": avg_mae, "r2_cv": avg_r2})

            results.append({
                "model": algo,
                "mae_cv": round(avg_mae, 2),
                "r2_cv": round(avg_r2, 4),
                "draw": self.draw,
                "region": self.region,
            })

        result_df = pd.DataFrame(results).sort_values("mae_cv")
        logger.success(f"Best model: {result_df.iloc[0]['model']} (MAE={result_df.iloc[0]['mae_cv']:.1f})")
        return result_df

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Save trained model to disk."""
        if self.model is None:
            raise RuntimeError("No model trained. Call train() first.")

        save_dir = output_dir or settings.MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / f"{self.model_name}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "features": self.feature_names,
                "params": self.best_params,
                "version": "v2.0.0",
                "draw": self.draw,
                "region": self.region,
                "model_type": self.model_type,
            }, f)

        logger.info(f"Model saved: {model_path}")
        return model_path

    # ── Internals ────────────────────────────────────────────────────────────

    def _walk_forward_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict,
        n_splits: int,
        model_type: Optional[str] = None,
    ) -> List[Dict]:
        """Walk-forward (time-series) cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            m = self._build_model(params, model_type or self.model_type)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_val)

            fold_result = {
                "fold": fold + 1,
                "mae": mean_absolute_error(y_val, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                "r2": r2_score(y_val, y_pred),
                "val_size": len(val_idx),
            }
            cv_results.append(fold_result)
            logger.debug(
                f"  Fold {fold+1}/{n_splits}: "
                f"MAE={fold_result['mae']:.1f}, R²={fold_result['r2']:.4f}"
            )

        return cv_results

    def _build_model(self, params: Dict, model_type: Optional[str] = None):
        """Instantiate the correct model class."""
        mt = model_type or self.model_type
        if mt == "lightgbm":
            return LGBMRegressor(**params)
        elif mt == "xgboost" and _HAS_XGB:
            return xgb.XGBRegressor(**params, verbosity=0)
        elif mt == "gradient_boosting":
            gb_params = {k: v for k, v in params.items()
                        if k in ["n_estimators", "learning_rate", "max_depth", "random_state"]}
            return GradientBoostingRegressor(**gb_params)
        else:
            raise ValueError(f"Unknown model_type: {mt}")

    def _default_params(self, model_type: Optional[str] = None) -> Dict:
        mt = model_type or self.model_type
        if mt == "lightgbm":
            return {
                "n_estimators": settings.N_ESTIMATORS,
                "learning_rate": settings.LEARNING_RATE,
                "max_depth": settings.MAX_DEPTH,
                "num_leaves": settings.NUM_LEAVES,
                "min_child_samples": settings.MIN_CHILD_SAMPLES,
                "subsample": settings.SUBSAMPLE,
                "colsample_bytree": settings.COLSAMPLE_BYTREE,
                "random_state": settings.RANDOM_SEED,
                "verbose": -1,
            }
        elif mt == "xgboost":
            return {
                "n_estimators": settings.N_ESTIMATORS,
                "learning_rate": settings.LEARNING_RATE,
                "max_depth": settings.MAX_DEPTH,
                "random_state": settings.RANDOM_SEED,
            }
        else:  # gradient_boosting
            return {
                "n_estimators": 150,
                "learning_rate": settings.LEARNING_RATE,
                "max_depth": 5,
                "random_state": settings.RANDOM_SEED,
            }
