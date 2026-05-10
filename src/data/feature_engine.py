"""
Feature engineering for BWF ranking time-series data.
Creates lag features, rolling statistics, and trend indicators
used by LightGBM and LSTM models.
Refactored from: prepare_ml_dataset.py (original notebook)
"""
from typing import List, Optional

import pandas as pd
import numpy as np

from src.config import settings
from src.utils.logger import logger


class FeatureEngineer:
    """
    Generates ML-ready features from cleaned BWF ranking data.

    Features created:
        - Lag features      : points at t-1, t-3, t-6, t-12 months
        - Rolling stats     : mean, std, min, max over 3/6/12-month windows
        - Trend features    : rank change, points change rate
        - Momentum features : consecutive months in top-N
        - Seasonality       : month of year (sin/cos encoded)

    Usage:
        fe = FeatureEngineer()
        features_df = fe.transform(clean_df, draw="MS", region="Global")
    """

    LAG_PERIODS: List[int] = [1, 3, 6, 12]
    ROLLING_WINDOWS: List[int] = [3, 6, 12]

    def __init__(self, lookback: int = None) -> None:
        self.lookback = lookback or settings.LOOKBACK_WINDOW

    def transform(
        self,
        df: pd.DataFrame,
        draw: str,
        region: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate feature set for a specific draw (and optional region).

        Args:
            df: Preprocessed BWF DataFrame (must have 'date', 'draw', 'player_id', 'points', 'rank').
            draw: One of MS / WS / MD / WD / XD.
            region: Optional continent filter (Asia / Europe / Global).

        Returns:
            DataFrame with feature columns + target column 'points'.
        """
        logger.info(f"Engineering features for draw={draw}, region={region or 'All'}")

        subset = df[df["draw"] == draw].copy()

        # 'Global' = no region filter; otherwise filter by region column
        if region and region != "Global" and "region" in df.columns:
            subset = subset[subset["region"] == region]

        if subset.empty:
            logger.warning(f"No data for draw={draw}, region={region}. Check region name matches CSV (Asia/Europe/Pan America/Africa).")
            return pd.DataFrame()

        # Sort by player then time for correct lag calculation
        sort_cols = [c for c in ["player_id", "date"] if c in subset.columns]
        subset = subset.sort_values(sort_cols)

        # Per-player feature generation
        result_parts = []
        for _, player_df in subset.groupby("player_id", group_keys=False):
            if len(player_df) > self.LAG_PERIODS[-1]: # Need enough data for lags
                result_parts.append(self._add_player_features(player_df.copy()))
        
        if not result_parts:
            logger.warning(f"No players in {draw}/{region} have enough history (>{self.LAG_PERIODS[-1]} months) to generate features.")
            return pd.DataFrame()

        result = pd.concat(result_parts, ignore_index=True)

        # Global features (rank relative to field)
        result = self._add_global_features(result)

        # Seasonality encoding (month column already in CSV)
        result = self._add_seasonality(result)

        # Drop rows with NaN from lag lookback period
        result = result.dropna(subset=self.feature_columns, how="any")
        logger.info(f"Features ready: {result.shape} for draw={draw}, region={region}")
        return result.reset_index(drop=True)

    # ── Feature builders ─────────────────────────────────────────────────────

    def _add_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag and rolling features per player."""
        df = df.sort_values("date")  # ensure chronological order within player

        for lag in self.LAG_PERIODS:
            df[f"points_lag_{lag}"] = df["points"].shift(lag)
            df[f"rank_lag_{lag}"] = df["rank"].shift(lag)

        for window in self.ROLLING_WINDOWS:
            df[f"points_roll_mean_{window}"] = df["points"].shift(1).rolling(window).mean()
            df[f"points_roll_std_{window}"] = df["points"].shift(1).rolling(window).std()
            df[f"points_roll_max_{window}"] = df["points"].shift(1).rolling(window).max()

        # Trend: point change from previous month
        df["points_delta_1m"] = df["points"].diff(1)
        df["points_delta_3m"] = df["points"].diff(3)
        df["points_pct_change_3m"] = df["points"].pct_change(3).replace([np.inf, -np.inf], 0)

        # Momentum: months consecutively ranked in top 10
        df["in_top10"] = (df["rank"] <= 10).astype(int)
        streak = []
        count = 0
        for val in df["in_top10"]:
            if val == 1:
                count += 1
            else:
                count = 0
            streak.append(count)
        df["top10_streak"] = streak

        return df

    def _add_global_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Field-relative rank features (per date)."""
        monthly = df.groupby("date")["points"].transform
        df["points_vs_field_mean"] = df["points"] / (monthly("mean") + 1e-6)
        df["points_vs_field_max"] = df["points"] / (monthly("max") + 1e-6)
        return df

    def _add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode month as cyclical sin/cos features."""
        if "month" not in df.columns:
            return df
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

    @property
    def feature_columns(self) -> List[str]:
        """List of all feature column names generated by this class."""
        cols = []
        for lag in self.LAG_PERIODS:
            cols += [f"points_lag_{lag}", f"rank_lag_{lag}"]
        for w in self.ROLLING_WINDOWS:
            cols += [f"points_roll_mean_{w}", f"points_roll_std_{w}", f"points_roll_max_{w}"]
        cols += [
            "points_delta_1m", "points_delta_3m", "points_pct_change_3m",
            "top10_streak",
            "points_vs_field_mean", "points_vs_field_max",
            "month_sin", "month_cos",
        ]
        return cols
