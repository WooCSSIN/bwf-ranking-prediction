"""
Data preprocessing for BWF ranking data.
Cleans raw data, handles missing values, and produces analysis-ready DataFrames.
Refactored from: bwf_official.py (original notebook)
"""
from typing import List, Optional

import pandas as pd
import numpy as np

from src.config import settings
from src.utils.logger import logger


class BWFPreprocessor:
    """
    Cleans and transforms raw BWF ranking data.

    Pipeline:
        1. Remove duplicates
        2. Handle missing values
        3. Filter valid draws & regions
        4. Normalize player names & country codes
        5. Sort chronologically

    Usage:
        preprocessor = BWFPreprocessor()
        clean_df = preprocessor.fit_transform(raw_df)
    """

    def __init__(
        self,
        draws: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
    ) -> None:
        self.draws = draws or settings.DRAWS
        self.regions = regions or settings.REGIONS

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")
        original_len = len(df)

        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df)
        df = self._filter_valid_draws(df)
        df = self._normalize_text_columns(df)
        df = self._add_time_features(df)
        df = self._sort(df)

        logger.info(
            f"Preprocessing complete: {original_len:,} -> {len(df):,} rows "
            f"({original_len - len(df):,} removed)"
        )
        return df

    # ── Pipeline steps ───────────────────────────────────────────────────────

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        subset = [c for c in ["player_id", "draw", "date"] if c in df.columns]
        df = df.drop_duplicates(subset=subset, keep="last")
        n_removed = n_before - len(df)
        if n_removed:
            logger.debug(f"Removed {n_removed:,} duplicate rows")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows without a rank or points (essential columns)
        essential = [c for c in ["rank", "points"] if c in df.columns]
        df = df.dropna(subset=essential)

        # Fill optional string columns
        for col in ["country_code", "player_name"]:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN")

        return df

    def _filter_valid_draws(self, df: pd.DataFrame) -> pd.DataFrame:
        if "draw" not in df.columns:
            return df
        before = len(df)
        df = df[df["draw"].isin(self.draws)]
        removed = before - len(df)
        if removed:
            logger.debug(f"Filtered {removed:,} rows with invalid draw codes")
        return df

    def _normalize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "player_name" in df.columns:
            df["player_name"] = df["player_name"].str.strip().str.title()
        if "country_code" in df.columns:
            df["country_code"] = df["country_code"].str.strip().str.upper()
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add year/month columns — skips if already present (CSV may already have them)."""
        if "date" not in df.columns:
            return df
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "year_month" not in df.columns:
            df["year_month"] = df["date"].dt.to_period("M").astype(str)
        return df

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [c for c in ["date", "draw", "rank"] if c in df.columns]
        return df.sort_values(sort_cols).reset_index(drop=True)
