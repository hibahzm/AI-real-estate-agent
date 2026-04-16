"""
supabase_logger.py — Saves queries and predictions to Supabase for analysis.

Every prediction and insight query is logged here. This serves two purposes:
  1. Assignment: shows Supabase integration
  2. Practical: lets you analyze which queries users ask and how the model performs

IMPORTANT: All logging is non-blocking and wrapped in try/except.
If Supabase is down or the key is wrong, the app still works — it just doesn't log.
Never let a logging failure break a user request.

Tables required in Supabase (see SETUP_GUIDE.md for SQL):
  - prediction_logs
  - insight_logs
"""

import logging
from typing import Optional
from supabase import create_client, Client

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level Supabase client — created once, reused for every log call
_client: Optional[Client] = None


def _get_client() -> Client:
    """
    Returns a Supabase client, creating it on first call.
    Using a module-level singleton avoids creating a new connection on every request.
    """
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


def log_prediction(
    user_query: str,
    extracted_features: Optional[dict],
    missing_features: list[str],
    predicted_price: Optional[float],
    interpretation: Optional[str],
    completeness_score: float,
    prompt_version: int,
    success: bool
) -> None:
    """
    Saves a prediction request and its result to the prediction_logs table.
    Silently swallows all errors — logging should never break the main flow.

    Args:
        user_query:          The original text the user typed
        extracted_features:  The features the LLM extracted (as a dict)
        missing_features:    Fields that were null after extraction
        predicted_price:     The ML model's output in dollars
        interpretation:      Stage 2 LLM explanation text
        completeness_score:  0.0–1.0, how complete the extraction was
        prompt_version:      1 or 2, for A/B testing analysis
        success:             Whether the full chain completed without errors
    """
    try:
        client = _get_client()
        client.table("prediction_logs").insert({
            "user_query":         user_query,
            "extracted_features": extracted_features,
            "missing_features":   missing_features,
            "predicted_price":    predicted_price,
            "interpretation":     interpretation,
            "completeness_score": completeness_score,
            "prompt_version":     prompt_version,
            "success":            success,
        }).execute()

        logger.debug(f"Logged prediction to Supabase — price=${predicted_price}")

    except Exception as e:
        # Log the error but don't re-raise — never break the request because of logging
        logger.warning(f"Supabase prediction logging failed (non-fatal): {e}")


def log_insight(
    user_query: str,
    insight_response: Optional[str],
    success: bool
) -> None:
    """
    Saves a market insight request and its response to the insight_logs table.

    Args:
        user_query:       The user's market question
        insight_response: The LLM's answer
        success:          Whether the insight was generated successfully
    """
    try:
        client = _get_client()
        client.table("insight_logs").insert({
            "user_query":       user_query,
            "insight_response": insight_response,
            "success":          success,
        }).execute()

        logger.debug(f"Logged insight to Supabase")

    except Exception as e:
        logger.warning(f"Supabase insight logging failed (non-fatal): {e}")


def fetch_recent_predictions(limit: int = 10) -> list[dict]:
    """
    Fetches the most recent prediction logs.
    Useful for admin dashboards or debugging.

    Returns an empty list if Supabase is unreachable.
    """
    try:
        client = _get_client()
        result = (
            client.table("prediction_logs")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.warning(f"Failed to fetch prediction logs: {e}")
        return []