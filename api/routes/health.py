from fastapi import APIRouter

from src.config import settings
from api.schemas.response import HealthResponse
from api.dependencies import get_model_cache_status

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and how many models are loaded in memory."""
    return HealthResponse(
        status="ok",
        api_version=settings.API_VERSION,
        model_count=get_model_cache_status()
    )
