"""
FastAPI application entrypoint for BWF Ranking Prediction.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.utils.logger import logger
from api.routes import health, rankings, predict

# Initialize app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
api_prefix = f"/api/{settings.API_VERSION}"
app.include_router(health.router, prefix=api_prefix)
app.include_router(rankings.router, prefix=api_prefix)
app.include_router(predict.router, prefix=api_prefix)

@app.on_event("startup")
async def startup_event():
    logger.info(f"🚀 Starting {settings.API_TITLE} - {settings.API_VERSION}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down API...")
