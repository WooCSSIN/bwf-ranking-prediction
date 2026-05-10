from fastapi import APIRouter, HTTPException, Query

from src.config import settings
from api.schemas.response import RankingResponse
from api.dependencies import get_predictor, get_latest_features

router = APIRouter(tags=["rankings"])

@router.get("/rankings/{draw}/{region}", response_model=RankingResponse)
async def get_rankings(
    draw: str, 
    region: str, 
    top_n: int = Query(10, ge=1, le=100, description="Number of top players to return")
):
    """
    Get Top-N predicted BWF badminton rankings for a specific draw and region.
    """
    draw = draw.upper()
    if draw not in settings.DRAWS:
        raise HTTPException(400, f"Invalid draw. Must be one of: {settings.DRAWS}")
    
    if region not in settings.REGIONS:
        raise HTTPException(400, f"Invalid region. Must be one of: {settings.REGIONS}")

    # Load model & features
    predictor = get_predictor(draw, region)
    features_df, player_meta, prediction_date = get_latest_features(draw, region)
    
    # Generate prediction
    rankings = predictor.get_top_n(features_df, player_meta, n=top_n)
    
    return RankingResponse(
        draw=draw,
        region=region,
        prediction_date=prediction_date,
        model_version=predictor.version,
        rankings=rankings
    )
