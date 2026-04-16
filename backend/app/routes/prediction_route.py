import logging
from fastapi import APIRouter, HTTPException, Query

from app.schemas.house_features_schema import HouseFeatures, PredictionRequest
from app.schemas.api_response_schema import (
    ExtractionResponse, FullPredictionResponse,
    ClassifyResponse,
)
from app.llm.intent_classifier import classify_intent, OTHER_RESPONSE_MESSAGE
from app.llm.feature_extractor_stage1 import extract_features
from app.llm.price_interpreter_stage2 import interpret_prediction
from app.ml.price_predictor import predict_price, get_training_stats

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


# ── /classify ──────────────────────────────────────────────────────────────

@router.post("/classify", response_model=ClassifyResponse)
async def classify_user_query(request: PredictionRequest):
    """
    Step 0: Classify the user's intent before doing any expensive work.

    Returns:
      intent == "prediction"  → proceed with /extract → /predict flow
      intent == "insights"    → call /insights directly
      intent == "other"       → show `message` to the user as a friendly reply
                                (greetings, off-topic, injection attempts, etc.)

    Example request:  {"user_query": "hi"}
    Example response: {"intent": "other", "confidence": 1.0,
                       "message": "I'm designed specifically for..."}
    """
    result = classify_intent(request.user_query)

    return ClassifyResponse(
        intent=result.intent,
        confidence=result.confidence,
        message=OTHER_RESPONSE_MESSAGE if result.intent == "other" else "",
    )


# ── /extract ───────────────────────────────────────────────────────────────

@router.post("/extract", response_model=ExtractionResponse)
async def extract_house_features(
    request: PredictionRequest,
    prompt_version: int = Query(
        default=1, ge=1, le=2, description="Prompt variant: 1 or 2"
    ),
):
    """
    Step 1: Parse natural language into structured house features.

    Returns extracted features + completeness metadata.
    If completeness_score < 1.0, the frontend shows a form for missing fields.

    Example request:
        {"user_query": "3-bedroom house with 2-car garage in a nice neighborhood"}

    Example response (partial):
        {
          "extraction": {
            "features": {"overall_quality": null, "gr_liv_area": 1500.0, "garage_cars": 2, ...},
            "extracted_fields": ["gr_liv_area", "garage_cars", "neighborhood"],
            "missing_fields": ["overall_quality", "total_basement_sf", ...],
            "completeness_score": 0.3,
            "notes": "Assumed 2-car garage from 'big garage'"
          },
          "success": true
        }
    """
    try:
        extraction = extract_features(
            user_query=request.user_query,
            prompt_version=prompt_version,
        )
        return ExtractionResponse(extraction=extraction, success=True)

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}"
        )


# ── /predict ───────────────────────────────────────────────────────────────

@router.post("/predict", response_model=FullPredictionResponse)
async def predict_house_price(features: HouseFeatures):
    """
    Step 2: Run the full chain — features → ML price → LLM interpretation.

    Expects ALL 10 features filled in (call /extract first, let user fill gaps).

    Example request:
        {
          "overall_quality": 7, "gr_liv_area": 1500.0, "garage_cars": 2,
          "total_basement_sf": 800.0, "full_bath_count": 2, "year_built": 2000,
          "lot_area": 8500.0, "neighborhood": "CollgCr",
          "exter_qual": "Gd", "kitchen_qual": "Gd"
        }
    """
    # Validate completeness
    missing = features.get_missing_fields()
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All 10 features must be filled before prediction",
                "missing_fields": missing,
                "hint": "Call /extract first, then fill missing fields in the UI",
            },
        )

    try:
        # Step A: ML prediction
        features_dict = features.to_model_input()
        predicted_price = predict_price(features_dict)

        # Step B: Stage 2 LLM interpretation
        training_stats = get_training_stats()
        interpretation = interpret_prediction(
            features=features,
            predicted_price=predicted_price,
            training_stats=training_stats,
        )

        # Step C: Market context for the UI charts/metrics
        market_context = {
            "median_price":      training_stats.get("median_price"),
            "price_p10":         training_stats.get("price_p10"),
            "price_p90":         training_stats.get("price_p90"),
            "neighborhood_avg":  training_stats.get(
                "avg_price_by_neighborhood", {}
            ).get(features.neighborhood, training_stats.get("median_price")),
        }

        # Step D: Log to Supabase (disabled for demo — uncomment to enable analytics)
        # log_prediction(
        #     user_query=str(features_dict),
        #     extracted_features=features_dict,
        #     missing_features=[],
        #     predicted_price=predicted_price,
        #     interpretation=interpretation,
        #     completeness_score=1.0,
        #     prompt_version=1,
        #     success=True,
        # )

        return FullPredictionResponse(
            features_used=features,
            predicted_price=predicted_price,
            interpretation=interpretation,
            market_context=market_context,
            success=True,
        )

    except HTTPException:
        raise  # re-raise 422 validation errors as-is

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        # log_prediction(
        #     user_query=str(features.model_dump()),
        #     extracted_features=features.model_dump(),
        #     missing_features=[],
        #     predicted_price=None,
        #     interpretation=None,
        #     completeness_score=1.0,
        #     prompt_version=1,
        #     success=False,
        # )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )