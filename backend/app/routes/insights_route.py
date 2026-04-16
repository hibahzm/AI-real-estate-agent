"""
insights_route.py — FastAPI route for market insight queries (BONUS feature).

This handles questions like:
  - "What are the most expensive neighborhoods?"
  - "How does garage size affect home value?"
  - "Which areas offer the best value?"

Instead of running an ML prediction, this feeds precomputed statistics from
training_statistics.json to the LLM and asks it to narrate real findings.

Every number in the response traces back to the actual training data.
"""

import logging
from fastapi import APIRouter, HTTPException
from openai import OpenAI, APIError

from app.schemas.house_features_schema import InsightRequest
from app.schemas.api_response_schema import InsightResponse
from app.ml.price_predictor import get_training_stats
from app.database.supabase_logger import log_insight
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Market Insights"])

INSIGHTS_SYSTEM_PROMPT = """You are a data-driven real estate market analyst.
You have access to statistics from 2,900+ home sales in Ames, Iowa (2006-2010).
Answer the user's question using ONLY the provided statistics — never invent numbers.
Be specific, cite actual figures, and give actionable insights.
Write in 3-5 clear sentences. Format dollar amounts with commas. Be conversational."""

INSIGHTS_USER_PROMPT = """
User Question: {user_query}

Real Ames Housing Market Statistics:
- Median sale price: ${median_price:,}
- Price range (10th to 90th percentile): ${p10:,} – ${p90:,}
- Total homes in dataset: {total_rows:,}

Price by Neighborhood (average):
{neighborhood_prices}

Price by Overall Quality (1-10 scale):
{quality_prices}

Price by House Style:
{style_prices}

Top 5 most expensive neighborhoods: {top_expensive}
Top 5 most affordable neighborhoods: {top_affordable}

Median price per sqft: ${price_per_sqft:.0f}
Price per sqft range (P25–P75): ${psf_p25:.0f} – ${psf_p75:.0f}

Top features that drive price (by model importance):
{feature_importance}

Answer the user's question using these statistics now:"""


def _format_dict_for_prompt(d: dict, prefix: str = "  ", value_format: str = "${:,}") -> str:
    """Formats a dict as a readable list for the LLM prompt."""
    lines = []
    for k, v in sorted(d.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0):
        if isinstance(v, (int, float)):
            lines.append(f"{prefix}{k}: {value_format.format(int(v))}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines[:20])   # cap at 20 items to keep prompt size reasonable


@router.post("/insights", response_model=InsightResponse)
async def get_market_insight(request: InsightRequest):
    """
    Answers a market question by grounding the LLM response in real training data statistics.

    Example requests:
        {"user_query": "What are the most expensive neighborhoods in Ames?"}
        {"user_query": "Does having a garage significantly increase home value?"}
        {"user_query": "What quality rating gives the best value for money?"}

    Example response:
        {
          "query": "What are the most expensive neighborhoods?",
          "insight": "The most exclusive neighborhoods in Ames are NridgHt ($316,270 avg),
                      StoneBr ($310,499 avg), and NoRidge ($307,875 avg)...",
          "success": true
        }
    """
    training_stats = get_training_stats()
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        # Format the statistics for the prompt
        neighborhood_prices = _format_dict_for_prompt(
            training_stats.get("avg_price_by_neighborhood", {})
        )
        quality_prices = _format_dict_for_prompt(
            {f"Quality {k}": v for k, v in training_stats.get("avg_price_by_quality", {}).items()}
        )
        style_prices = _format_dict_for_prompt(
            training_stats.get("avg_price_by_house_style", {})
        )
        feature_importance = "\n".join([
            f"  {feat}: {imp:.1%}"
            for feat, imp in list(training_stats.get("feature_importance", {}).items())[:8]
        ])

        user_prompt = INSIGHTS_USER_PROMPT.format(
            user_query=request.user_query,
            median_price=training_stats.get("median_price", 163000),
            p10=training_stats.get("price_p10", 88000),
            p90=training_stats.get("price_p90", 278000),
            total_rows=training_stats.get("total_rows", 2930),
            neighborhood_prices=neighborhood_prices,
            quality_prices=quality_prices,
            style_prices=style_prices,
            top_expensive=", ".join(training_stats.get("top_5_most_expensive_neighborhoods", [])),
            top_affordable=", ".join(training_stats.get("top_5_most_affordable_neighborhoods", [])),
            price_per_sqft=training_stats.get("price_per_sqft_median", 112),
            psf_p25=training_stats.get("price_per_sqft_p25", 85),
            psf_p75=training_stats.get("price_per_sqft_p75", 140),
            feature_importance=feature_importance,
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        insight_text = response.choices[0].message.content.strip()

        # Log to Supabase
        log_insight(
            user_query=request.user_query,
            insight_response=insight_text,
            success=True
        )

        return InsightResponse(
            query=request.user_query,
            insight=insight_text,
            success=True
        )

    except APIError as e:
        logger.error(f"Insights: OpenAI API error: {e}")
        log_insight(request.user_query, None, False)
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")

    except Exception as e:
        logger.error(f"Insights route failed: {e}", exc_info=True)
        log_insight(request.user_query, None, False)
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")