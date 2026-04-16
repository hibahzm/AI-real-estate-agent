"""
prediction_route.py — FastAPI routes for the price prediction pipeline.

Two endpoints:
  POST /api/v1/extract   → Stage 1 only: text → features (with completeness info)
  POST /api/v1/predict   → Full chain: features → ML price → Stage 2 interpretation

Why two separate endpoints?
  The UI calls /extract first, shows the user what was found and what's missing,
  lets the user fill in gaps, then calls /predict with complete features.
  This way the user always reviews the extracted features before a prediction runs.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.schemas.house_features_schema import (
    HouseFeatures,
    PredictionRequest,
)
from app.schemas.api_response_schema import ExtractionResponse, FullPredictionResponse
from app.llm.feature_extractor_stage1 import extract_features
from app.llm.price_interpreter_stage2 import interpret_prediction
from app.ml.price_predictor import predict_price, get_training_stats
from app.database.supabase_logger import log_prediction

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post("/extract", response_model=ExtractionResponse)
async def extract_house_features(
    request: PredictionRequest,
    prompt_version: int = Query(default=1, ge=1, le=2, description="Prompt variant: 1 or 2")
):
    """
    Stage 1 of the pipeline: parse natural language into house features.

    Returns extracted features + completeness info.
    If completeness_score < 1.0, the frontend should show a form for missing fields.

    Example request:
        {"user_query": "3 bedroom house with big garage in a good neighborhood"}

    Example response (partial):
        {
          "extraction": {
            "features": {"bedroom_count": 3, "garage_area": 600.0, "neighborhood": "CollgCr", ...},
            "extracted_fields": ["bedroom_count", "garage_area", "neighborhood"],
            "missing_fields": ["overall_quality", "gr_liv_area", ...],
            "completeness_score": 0.3,
            "notes": "Assumed large garage = ~600 sqft"
          },
          "success": true
        }
    """
    try:
        extraction = extract_features(
            user_query=request.user_query,
            prompt_version=prompt_version
        )

        return ExtractionResponse(
            extraction=extraction,
            success=True
        )

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}"
        )


@router.post("/predict", response_model=FullPredictionResponse)
async def predict_house_price(features: HouseFeatures):
    """
    Full prediction chain: complete features → ML price → LLM interpretation.

    This endpoint expects ALL 10 features to be filled in.
    The frontend should call /extract first, let the user fill gaps, then call this.

    Example request:
        {
          "overall_quality": 7,
          "gr_liv_area": 1500.0,
          "total_basement_sf": 800.0,
          "garage_area": 400.0,
          "year_built": 2000,
          "lot_area": 8000.0,
          "bedroom_count": 3,
          "full_bath_count": 2,
          "neighborhood": "CollgCr",
          "house_style": "2Story"
        }
    """
    # Validate: all 10 features must be present
    missing = features.get_missing_fields()
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All features must be filled before prediction",
                "missing_fields": missing,
                "hint": "Call /extract first, then fill missing fields in the UI"
            }
        )

    interpretation = "Interpretation unavailable"
    predicted_price = None

    try:
        # Step 1: ML prediction
        # to_model_input() returns the original sklearn column names
        # (OverallQual, GrLivArea, …) that the pipeline was trained on
        features_dict = features.to_model_input()
        predicted_price = predict_price(features_dict)

        # Step 2: Stage 2 LLM interpretation
        training_stats = get_training_stats()
        interpretation = interpret_prediction(
            features=features,
            predicted_price=predicted_price,
            training_stats=training_stats
        )

        # Step 3: Build market context for the UI
        market_context = {
            "median_price":    training_stats.get("median_price"),
            "price_p10":       training_stats.get("price_p10"),
            "price_p90":       training_stats.get("price_p90"),
            "neighborhood_avg": training_stats.get("avg_price_by_neighborhood", {}).get(
                features.neighborhood, training_stats.get("median_price")
            ),
        }

        # Step 4: Log to Supabase (non-blocking)
        log_prediction(
            user_query=str(features_dict),
            extracted_features=features_dict,
            missing_features=[],
            predicted_price=predicted_price,
            interpretation=interpretation,
            completeness_score=1.0,
            prompt_version=1,
            success=True
        )

        return FullPredictionResponse(
            features_used=features,
            predicted_price=predicted_price,
            interpretation=interpretation,
            market_context=market_context,
            success=True
        )

    except HTTPException:
        raise   # re-raise validation errors as-is

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)

        # Log the failure
        log_prediction(
            user_query=str(features.model_dump()),
            extracted_features=features.model_dump(),
            missing_features=[],
            predicted_price=None,
            interpretation=None,
            completeness_score=1.0,
            prompt_version=1,
            success=False
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )