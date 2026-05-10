"""
Evaluation metrics for BWF ranking prediction models.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute regression metrics for ranking point prediction.

    Args:
        y_true: Actual points values.
        y_pred: Predicted points values.
        prefix: Optional prefix for metric keys (e.g. 'train_', 'val_').

    Returns:
        Dictionary of metric name → value.
    """
    p = f"{prefix}" if prefix else ""
    metrics = {
        f"{p}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{p}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{p}r2": float(r2_score(y_true, y_pred)),
        f"{p}mape": float(_mape(y_true, y_pred)),
    }

    logger.info(
        f"Metrics [{prefix.strip('_') or 'eval'}]: "
        f"MAE={metrics[f'{p}mae']:.1f}, "
        f"RMSE={metrics[f'{p}rmse']:.1f}, "
        f"R²={metrics[f'{p}r2']:.4f}, "
        f"MAPE={metrics[f'{p}mape']:.2f}%"
    )
    return metrics


def rank_accuracy(
    y_true_ranks: np.ndarray,
    y_pred_ranks: np.ndarray,
    top_n: int = 10,
) -> Dict[str, float]:
    """
    Compute ranking-specific accuracy metrics.

    Args:
        y_true_ranks: True player ranks.
        y_pred_ranks: Predicted player ranks.
        top_n:        Threshold for 'in top-N' accuracy.

    Returns:
        Dictionary with rank_corr (Spearman) and top_n_precision.
    """
    from scipy.stats import spearmanr

    corr, pval = spearmanr(y_true_ranks, y_pred_ranks)

    # Top-N precision: % of truly top-N players that are also in predicted top-N
    true_topn = set(np.argsort(y_true_ranks)[:top_n])
    pred_topn = set(np.argsort(y_pred_ranks)[:top_n])
    precision = len(true_topn & pred_topn) / top_n

    return {
        "spearman_corr": float(corr),
        "spearman_pval": float(pval),
        f"top{top_n}_precision": float(precision),
    }


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (ignores zero-valued actuals)."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
