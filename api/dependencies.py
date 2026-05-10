from typing import Dict, Tuple
from functools import lru_cache

from fastapi import HTTPException
import pandas as pd

from src.models.predictor import BWFPredictor
from src.data.loader import DataLoader
from src.data.preprocessor import BWFPreprocessor
from src.data.feature_engine import FeatureEngineer

# In-memory cache for loaded models
_model_cache: Dict[str, BWFPredictor] = {}
# Global cache for the base preprocessed dataframe
_global_clean_df: pd.DataFrame = None
# Cache for processed features
_feature_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}


def get_predictor(draw: str, region: str) -> BWFPredictor:
    """Load and cache the trained LightGBM model."""
    key = f"{draw}_{region}"
    if key not in _model_cache:
        try:
            predictor = BWFPredictor(draw=draw, region=region)
            predictor.load()
            _model_cache[key] = predictor
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, 
                detail=f"No trained model found for Draw={draw}, Region={region}. Please train it first."
            )
    return _model_cache[key]


@lru_cache(maxsize=10)
def get_latest_features(draw: str, region: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Get the latest engineered features to make predictions.
    Returns: (features_df, player_meta_df, prediction_date_str)
    """
    global _global_clean_df
    if _global_clean_df is None:
        loader = DataLoader()
        try:
            raw_df = loader.load("bwf_cleaned.csv")
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Data file bwf_cleaned.csv not found on server.")
        
        preprocessor = BWFPreprocessor()
        _global_clean_df = preprocessor.fit_transform(raw_df)

    clean_df = _global_clean_df
    
    # 3. Extract latest state (we only want the most recent date per player for prediction)
    latest_date = clean_df["date"].max()
    # Filter to only the latest date
    current_df = clean_df[clean_df["date"] == latest_date].copy()
    
    # 4. Feature engineering
    fe = FeatureEngineer()
    features_df = fe.transform(clean_df, draw=draw, region=region)
    
    # Filter features to only the latest date for actual inference
    latest_features = features_df[features_df["date"] == latest_date].copy()
    
    if latest_features.empty:
        raise HTTPException(status_code=404, detail="No feature data available to make predictions.")
        
    player_meta = latest_features[["player_id", "player_name", "country_code"]].copy()
    
    # In real world time-series, the prediction date is typically 1 period ahead of latest data
    next_date = latest_date + pd.DateOffset(months=1)
    prediction_date_str = next_date.strftime("%Y-%m-%d")
    
    return latest_features, player_meta, prediction_date_str

def get_model_cache_status() -> int:
    return len(_model_cache)
