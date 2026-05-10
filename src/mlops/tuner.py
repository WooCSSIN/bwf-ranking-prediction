"""
Hyperparameter tuning with Optuna for BWF LightGBM models.
Runs time-series cross-validated search and returns best params.
"""
from typing import Callable, Dict, Optional

import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from src.config import settings
from src.utils.logger import logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise ImportError("lightgbm required: pip install lightgbm")


class HyperparameterTuner:
    """
    Uses Optuna TPE sampler + TimeSeriesSplit to find optimal LightGBM params.

    Usage:
        tuner = HyperparameterTuner(draw="MS", region="Asia")
        best_params = tuner.optimize(X, y, n_trials=50)
    """

    def __init__(self, draw: str, region: str = "Global", n_splits: int = 5) -> None:
        self.draw = draw.upper()
        self.region = region
        self.n_splits = n_splits
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict] = None

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        timeout_seconds: Optional[int] = 300,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Run Optuna hyperparameter search.

        Args:
            X:               Feature matrix (sorted by time).
            y:               Target array.
            n_trials:        Number of Optuna trials.
            timeout_seconds: Max search time (seconds).
            callback:        Optional Optuna callback.

        Returns:
            Dictionary of best hyperparameters.
        """
        study_name = f"bwf_{self.draw}_{self.region}_tuning"
        logger.info(
            f"Starting Optuna study: {study_name} | "
            f"trials={n_trials}, timeout={timeout_seconds}s"
        )

        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize",       # Minimize MAE
            sampler=optuna.samplers.TPESampler(seed=settings.RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )

        objective = self._make_objective(X, y)
        callbacks = [self._logging_callback]
        if callback:
            callbacks.append(callback)

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            callbacks=callbacks,
            n_jobs=1,          # 1 for reproducibility; set >1 for speed
        )

        self.best_params = self.study.best_params
        logger.success(
            f"Best trial #{self.study.best_trial.number}: "
            f"MAE={self.study.best_value:.2f} | params={self.best_params}"
        )
        return self.best_params

    def get_study_summary(self) -> Dict:
        """Return a summary of the Optuna study results."""
        if self.study is None:
            return {}
        return {
            "n_trials": len(self.study.trials),
            "best_trial": self.study.best_trial.number,
            "best_mae": self.study.best_value,
            "best_params": self.study.best_params,
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _make_objective(self, X: np.ndarray, y: np.ndarray) -> Callable:
        """Build Optuna objective function with walk-forward CV."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": settings.RANDOM_SEED,
                "verbose": -1,
            }

            fold_maes = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

                y_pred = model.predict(X_val)
                fold_maes.append(mean_absolute_error(y_val, y_pred))

                # Pruning: report intermediate value
                trial.report(np.mean(fold_maes), step=fold)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return float(np.mean(fold_maes))

        return objective

    @staticmethod
    def _logging_callback(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        """Log every 10th trial to avoid log spam."""
        if trial.number % 10 == 0:
            logger.info(
                f"  Trial {trial.number:03d}: MAE={trial.value:.2f} "
                f"(best so far: {study.best_value:.2f})"
            )
