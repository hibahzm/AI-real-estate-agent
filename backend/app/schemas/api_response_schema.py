from pydantic import BaseModel
from typing import Optional
from app.schemas.house_features_schema import HouseFeatures, ExtractionResult


class ExtractionResponse(BaseModel):
    """Response from POST /api/v1/extract"""
    extraction: ExtractionResult
    success: bool
    error_message: Optional[str] = None


class FullPredictionResponse(BaseModel):
    """Response from POST /api/v1/predict"""
    features_used: HouseFeatures
    predicted_price: float
    interpretation: str
    market_context: dict
    success: bool
    error_message: Optional[str] = None


class InsightResponse(BaseModel):
    """Response from POST /api/v1/insights"""
    query: str
    insight: str
    success: bool
    error_message: Optional[str] = None


class IntentClassificationResult(BaseModel):
    """
    Internal schema — not returned directly by an endpoint.
    intent: "prediction" | "insights" | "other"
    """
    intent: str
    confidence: float
    reasoning: str = ""


class ClassifyResponse(BaseModel):
    """
    Response from POST /api/v1/classify
    The frontend calls this FIRST to decide which UI flow to show.
      intent == "prediction" → show extraction form flow
      intent == "insights"   → call /insights endpoint
      intent == "other"      → show the message field as a friendly reply
    """
    intent: str       # "prediction" | "insights" | "other"
    confidence: float
    message: str = "" # filled when intent == "other"


class HealthCheckResponse(BaseModel):
    """Response from GET /health"""
    status: str
    model_loaded: bool
    stats_loaded: bool
    message: str = ""