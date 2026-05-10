from fastapi import APIRouter, HTTPException

from api.schemas.request import PredictionRequest
from api.schemas.response import PredictionResponse
from api.dependencies import get_predictor, get_latest_features
from src.config import settings

router = APIRouter(tags=["predictions"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_player(request: PredictionRequest):
    """
    Predict future points for a specific player ID.
    """
    predictor = get_predictor(request.draw, request.region)
    features_df, player_meta, prediction_date = get_latest_features(request.draw, request.region)
    
    # Find the player's features
    player_features = features_df[features_df["player_id"] == request.player_id]
    
    if player_features.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Player ID {request.player_id} not found in the latest dataset for {request.draw}/{request.region}."
        )
    
    # Predict
    predicted_points = predictor.predict(player_features)[0]
    
    return PredictionResponse(
        player_id=request.player_id,
        draw=request.draw,
        region=request.region,
        prediction_date=prediction_date,
        predicted_points=float(predicted_points),
        model_version=predictor.version
    )
