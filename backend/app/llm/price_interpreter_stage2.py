"""
price_interpreter_stage2.py — Stage 2 of the LLM prompt chain.
"""

import logging
from openai import OpenAI, APIError, RateLimitError

from app.schemas.house_features_schema import HouseFeatures
from app.config import settings

logger = logging.getLogger(__name__)

STAGE2_SYSTEM_PROMPT = """You are a senior real estate market analyst in Ames, Iowa.
You receive structured property data, a machine learning price prediction, and market statistics.
Write a clear, specific, insightful 3-4 sentence market analysis.

Rules:
- Be specific with dollar amounts and percentages
- Identify the top 1-2 factors driving the price (quality, size, neighborhood, age)
- Compare to the neighborhood average and market median
- Say whether this is a deal, fair value, or premium pricing
- Do NOT just restate the prediction — add real insight
- Write in professional but conversational tone
- Never say "I" or "the model" — speak as a market analyst"""

STAGE2_USER_PROMPT = """
Property Being Analyzed:
- Overall Quality  : {overall_quality}/10
- Living Area      : {gr_liv_area:,.0f} sqft (above grade)
- Basement Area    : {total_basement_sf:,.0f} sqft
- Garage Capacity  : {garage_cars} car(s)
- Full Bathrooms   : {full_bath_count}
- Year Built       : {year_built}
- Lot Area         : {lot_area:,.0f} sqft
- Neighborhood     : {neighborhood}
- Exterior Quality : {exter_qual}  (Po=Poor → Ex=Excellent)
- Kitchen Quality  : {kitchen_qual}  (Po=Poor → Ex=Excellent)

ML Model Prediction: ${predicted_price:,.0f}

Market Context (Ames Housing, 2006-2010):
- Overall market median        : ${median_price:,.0f}
- Typical range (P10–P90)      : ${p10:,.0f} – ${p90:,.0f}
- Average price in {neighborhood}: ${neighborhood_avg:,.0f}
- Average price for quality {overall_quality}/10: ${quality_avg:,.0f}
- Median price per sqft        : ${price_per_sqft:.0f}/sqft
- This prediction vs median    : {pct_vs_median:+.1f}%

Write the 3-4 sentence analysis now:"""


def interpret_prediction(
    features: HouseFeatures,
    predicted_price: float,
    training_stats: dict
) -> str:
    client = OpenAI(api_key=settings.openai_api_key)

    neighborhood  = features.neighborhood   or "NAmes"
    quality       = features.overall_quality or 5

    neighborhood_avg = int(
        training_stats.get("avg_price_by_neighborhood", {}).get(
            neighborhood, training_stats.get("median_price", 160000)
        )
    )
    quality_avg = int(
        training_stats.get("avg_price_by_quality", {}).get(
            str(quality), training_stats.get("median_price", 160000)
        )
    )

    median_price  = training_stats.get("median_price", 160000)
    pct_vs_median = (predicted_price - median_price) / median_price * 100

    user_prompt = STAGE2_USER_PROMPT.format(
        overall_quality   = features.overall_quality   or 5,
        gr_liv_area       = features.gr_liv_area       or 0,
        total_basement_sf = features.total_basement_sf or 0,
        garage_cars       = features.garage_cars       or 0,
        full_bath_count   = features.full_bath_count   or 0,
        year_built        = features.year_built        or "Unknown",
        lot_area          = features.lot_area          or 0,
        neighborhood      = neighborhood,
        exter_qual        = features.exter_qual        or "TA",
        kitchen_qual      = features.kitchen_qual      or "TA",
        predicted_price   = predicted_price,
        median_price      = median_price,
        p10               = training_stats.get("price_p10",  88000),
        p90               = training_stats.get("price_p90", 278000),
        neighborhood_avg  = neighborhood_avg,
        quality_avg       = quality_avg,
        price_per_sqft    = training_stats.get("price_per_sqft_median", 112),
        pct_vs_median     = pct_vs_median,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()

    except (APIError, RateLimitError) as e:
        logger.error(f"Stage 2 API error: {e}")
        return _fallback_interpretation(predicted_price, median_price, neighborhood, neighborhood_avg)
    except Exception as e:
        logger.error(f"Stage 2 error: {e}", exc_info=True)
        return _fallback_interpretation(predicted_price, median_price, neighborhood, neighborhood_avg)


def _fallback_interpretation(predicted_price, median_price, neighborhood, neighborhood_avg):
    pct       = (predicted_price - median_price) / median_price * 100
    direction = "above" if pct > 0 else "below"
    vs_nbhd   = "above" if predicted_price > neighborhood_avg else "below"
    return (
        f"The estimated price for this property is ${predicted_price:,.0f}. "
        f"This is {abs(pct):.1f}% {direction} the Ames market median of ${median_price:,.0f}. "
        f"Compared to the {neighborhood} neighborhood average of ${neighborhood_avg:,.0f}, "
        f"this property is priced {vs_nbhd} average."
    )