"""
Centralized configuration management for BWF Ranking Prediction.
All settings are loaded from environment variables or .env file.
"""
from pathlib import Path
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # fallback for older pydantic


class Settings(BaseSettings):
    # ──────────────────────────────────────────
    # Project paths
    # ──────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    DATA_PREDICTIONS_DIR: Path = BASE_DIR / "data" / "predictions"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # ──────────────────────────────────────────
    # Model parameters
    # ──────────────────────────────────────────
    N_ESTIMATORS: int = 200
    LEARNING_RATE: float = 0.05
    MAX_DEPTH: int = 5
    NUM_LEAVES: int = 31
    MIN_CHILD_SAMPLES: int = 20
    SUBSAMPLE: float = 0.8
    COLSAMPLE_BYTREE: float = 0.8
    RANDOM_SEED: int = 42

    # ──────────────────────────────────────────
    # Forecast settings
    # ──────────────────────────────────────────
    FORECAST_MONTHS: int = 120          # Forecast horizon (e.g. to 2035)
    BASE_YEAR: int = 2024               # Last known data year
    TARGET_YEAR: int = 2035             # Prediction target year
    LOOKBACK_WINDOW: int = 12           # Months of history used as features

    # ──────────────────────────────────────────
    # BWF domain settings
    # ──────────────────────────────────────────
    DRAWS: List[str] = ["MS", "WS", "MD", "WD", "XD"]
    # 'Global' is a special value meaning no region filter
    REGIONS: List[str] = ["Global", "Asia", "Europe", "Pan America", "Africa"]
    TOP_N_DEFAULT: int = 10

    # ──────────────────────────────────────────
    # API settings (Phase 3)
    # ──────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_VERSION: str = "v1"
    API_TITLE: str = "BWF Badminton Ranking Prediction API"
    API_DESCRIPTION: str = (
        "REST API for predicting BWF badminton world rankings up to 2035 "
        "using hybrid LightGBM + LSTM models."
    )

    # ──────────────────────────────────────────
    # MLflow settings (Phase 2)
    # ──────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_PREFIX: str = "bwf"

    # ──────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance — import this everywhere
settings = Settings()
