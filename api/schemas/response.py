from typing import List, Optional
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    api_version: str
    model_count: int

class PlayerRanking(BaseModel):
    rank: int
    player_name: str
    country_code: str
    predicted_points: float

class RankingResponse(BaseModel):
    draw: str
    region: str
    prediction_date: str
    model_version: str
    rankings: List[PlayerRanking]

class PredictionResponse(BaseModel):
    player_id: int
    draw: str
    region: str
    prediction_date: str
    predicted_points: float
    model_version: str
