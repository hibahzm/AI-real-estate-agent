"""
intent_classifier.py — Decides what the user wants before anything else runs.

Two possible intents:
  - "prediction" → user wants to estimate a property's price
  - "insights"   → user wants to understand market trends, neighborhood info, etc.

This is the BONUS feature: the app automatically routes the query to the right pipeline.
"""

import logging
from openai import OpenAI, APIError

from app.schemas.api_response_schema import IntentClassificationResult
from app.config import settings

logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """You are a routing assistant for a real estate AI application.
Classify the user's query into exactly one of two categories:

"prediction" — the user wants to estimate the price of a specific property.
Examples:
  - "How much would a 3-bedroom house cost?"
  - "What's the price of a ranch with a 2-car garage?"
  - "Estimate the value of a 1500 sqft colonial in a nice area"
  - "What's my house worth?"

"insights" — the user wants market information, trends, comparisons, or analysis.
Examples:
  - "What are the most expensive neighborhoods?"
  - "How does garage size affect home price?"
  - "Which neighborhoods are undervalued?"
  - "What's the average price per sqft?"
  - "Show me market trends"
  - "What's a good area to buy in?"

Respond with JSON only:
{
  "intent": "prediction" or "insights",
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence explanation"
}"""


def classify_intent(user_query: str) -> IntentClassificationResult:
    """
    Classifies the user's query as either a prediction request or an insights request.

    Args:
        user_query: The raw user input

    Returns:
        IntentClassificationResult with intent = "prediction" or "insights"
        Falls back to "prediction" if classification fails (safest default).
    """
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLASSIFIER_PROMPT},
                {"role": "user",   "content": user_query}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,   # deterministic — classification should never be random
            max_tokens=100
        )

        import json
        data = json.loads(response.choices[0].message.content)

        intent = data.get("intent", "prediction")
        if intent not in ("prediction", "insights"):
            intent = "prediction"   # safe default for unknown values

        result = IntentClassificationResult(
            intent=intent,
            confidence=float(data.get("confidence", 0.9)),
            reasoning=data.get("reasoning", "")
        )

        logger.info(f"Intent classified: {result.intent} (confidence={result.confidence:.0%})")
        return result

    except (APIError, Exception) as e:
        logger.warning(f"Intent classification failed, defaulting to 'prediction': {e}")
        # Default to prediction — it's the primary feature and safer fallback
        return IntentClassificationResult(
            intent="prediction",
            confidence=0.5,
            reasoning="Classification failed — defaulting to prediction"
        )