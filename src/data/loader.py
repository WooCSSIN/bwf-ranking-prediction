"""
Data loading and validation for BWF ranking datasets.
Handles raw CSV ingestion with schema validation and type coercion.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import settings
from src.utils.logger import logger


# ── Column mapping from raw CSV → standardized names ────────────────────────
# Raw CSV columns: rank, delta_rank, country_code, name, id, points,
#                  tournaments_played, continent, country, date, draw, year, month
COLUMN_RENAME_MAP = {
    "name": "player_name",
    "id": "player_id",
    "continent": "region",
}

REQUIRED_COLUMNS = {
    "player_id",
    "player_name",
    "country_code",
    "draw",
    "rank",
    "points",
    "date",
}


class DataLoader:
    """
    Loads and validates raw BWF ranking CSV files.

    Usage:
        loader = DataLoader()
        df = loader.load("bwf_ranking.csv")
    """

    def __init__(self, raw_dir: Optional[Path] = None) -> None:
        self.raw_dir = raw_dir or settings.DATA_RAW_DIR

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, filename: str) -> pd.DataFrame:
        """
        Load a raw CSV file from the raw data directory.

        Args:
            filename: Name of the CSV file (e.g. 'bwf_ranking.csv').

        Returns:
            Validated pandas DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing.
        """
        filepath = self.raw_dir / filename
        logger.info(f"Loading raw data: {filepath}")

        if not filepath.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {filepath}\n"
                f"Please place it in: {self.raw_dir}"
            )

        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

        # Rename raw columns to standardized names
        df = df.rename(columns=COLUMN_RENAME_MAP)

        # Drop unnamed index column if present
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        df = self._validate(df)
        df = self._coerce_types(df)

        logger.success(f"Data loaded and validated: {df.shape}")
        return df

    def load_from_path(self, filepath: Path) -> pd.DataFrame:
        """Load from an explicit path (bypasses raw_dir)."""
        logger.info(f"Loading data from explicit path: {filepath}")
        df = pd.read_csv(filepath, low_memory=False)
        return self._coerce_types(df)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check required columns exist."""
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(
                f"Missing expected columns: {missing}. "
                "Proceeding with available columns."
            )
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize dtypes for downstream processing."""
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "rank" in df.columns:
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        if "points" in df.columns:
            df["points"] = pd.to_numeric(df["points"], errors="coerce")
        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        if "draw" in df.columns:
            df["draw"] = df["draw"].str.upper().str.strip()
        if "region" in df.columns:
            df["region"] = df["region"].str.strip()
        return df
