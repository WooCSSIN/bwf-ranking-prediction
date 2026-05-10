"""
Model inference: load a trained model and generate predictions.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logger import logger


class BWFPredictor:
    """
    Loads a saved LightGBM model and generates BWF ranking predictions.

    Usage:
        predictor = BWFPredictor(draw="MS", region="Asia")
        predictor.load()
        top10 = predictor.get_top_n(n=10)
    """

    def __init__(self, draw: str, region: str = "Global") -> None:
        self.draw = draw.upper()
        self.region = region
        self.model_name = f"lightgbm_{self.draw}_{self.region}".lower()
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.version: str = "unknown"

    def load(self, model_dir: Optional[Path] = None) -> "BWFPredictor":
        """Load model from disk."""
        model_dir = model_dir or settings.MODELS_DIR
        model_path = model_dir / f"{self.model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                f"Run training first: python scripts/train.py --draw {self.draw} --region {self.region}"
            )

        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload["model"]
        self.feature_names = payload.get("features", [])
        self.version = payload.get("version", "v1.0.0")
        logger.info(f"Loaded model: {self.model_name} (version={self.version})")
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Run inference on a feature DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        X = features[self.feature_names].values
        return self.model.predict(X)

    def get_top_n(
        self,
        features_df: pd.DataFrame,
        player_meta: pd.DataFrame,
        n: int = None,
    ) -> List[Dict]:
        """
        Predict points for all players and return ranked top-N list.

        Args:
            features_df:  DataFrame with feature columns.
            player_meta:  DataFrame with player_id, player_name, country_code.
            n:            Number of top players to return.

        Returns:
            List of dicts with rank, player_name, country_code, predicted_points.
        """
        n = n or settings.TOP_N_DEFAULT
        predicted_points = self.predict(features_df)

        result_df = player_meta.copy()
        result_df["predicted_points"] = predicted_points
        result_df = result_df.sort_values("predicted_points", ascending=False).head(n)
        result_df["rank"] = range(1, len(result_df) + 1)

        return result_df[["rank", "player_name", "country_code", "predicted_points"]].to_dict("records")
