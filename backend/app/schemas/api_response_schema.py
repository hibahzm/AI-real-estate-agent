"""
api_response_schema.py — Shapes of the JSON responses our API sends back.

Every endpoint returns one of these schemas so the frontend always knows
exactly what shape to expect.
"""

from pydantic import BaseModel
from typing import Optional
from app.schemas.house_features_schema import HouseFeatures, ExtractionResult


class ExtractionResponse(BaseModel):
    """
    Response from POST /api/v1/extract
    Stage 1 result — what the LLM found in the user's text.
    If completeness_score < 1.0, the frontend shows a form for missing fields.
    """
    extraction: ExtractionResult
    success: bool
    error_message: Optional[str] = None


class FullPredictionResponse(BaseModel):
    """
    Response from POST /api/v1/predict
    The full chain result: features → ML price → LLM interpretation.
    """
    # The features that were used to make the prediction
    features_used: HouseFeatures

    # Dollar amount from the ML model
    predicted_price: float

    # Human-readable explanation from Stage 2 LLM
    interpretation: str

    # Summary stats included for the frontend to display
    market_context: dict  # median_price, price_p10, price_p90, etc.

    success: bool
    error_message: Optional[str] = None


class InsightResponse(BaseModel):
    """
    Response from POST /api/v1/insights
    The LLM's answer to a market question, grounded in real training data.
    """
    query: str
    insight: str
    success: bool
    error_message: Optional[str] = None


class IntentClassificationResult(BaseModel):
    """
    Internal schema used by the intent classifier.
    Not returned directly by an endpoint — used internally.
    """
    intent: str              # "prediction" or "insights"
    confidence: float        # 0.0 to 1.0
    reasoning: str = ""      # why the classifier chose this intent


class HealthCheckResponse(BaseModel):
    """
    Response from GET /health
    Used to verify the service is up and the model is loaded.
    """
    status: str              # "ok" or "error"
    model_loaded: bool
    stats_loaded: bool
    message: str = ""