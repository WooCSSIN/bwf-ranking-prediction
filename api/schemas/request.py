from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    player_id: int = Field(..., description="BWF Player ID")
    draw: str = Field(..., pattern="^(MS|WS|MD|WD|XD)$", description="Draw category")
    region: str = Field("Global", pattern="^(Global|Asia|Europe|Pan America|Africa)$", description="Region")
