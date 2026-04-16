import json
import logging
from typing import Optional
from openai import OpenAI, APIError, RateLimitError

from app.schemas.house_features_schema import HouseFeatures, ExtractionResult
from app.config import settings

logger = logging.getLogger(__name__)

# ── Valid categorical values — confirmed from actual train split output ──
# After applying threshold=20 on the training set, exactly 21 neighborhoods
# remain as distinct categories. Blmngtn (~17 in train) and NPkVill (~14)
# also fell below 20 and were grouped into Other.
# Result: 22 valid values (21 neighborhoods + "Other")
VALID_NEIGHBORHOODS = [
    "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
    "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes", "NoRidge", "NridgHt", "NWAmes", "OldTown",
    "Other",    # Blmngtn, NPkVill, Blueste, Greens, GrnHill, Landmrk
    "Sawyer", "SawyerW", "SWISU", "Somerst", "StoneBr",
    "Timber", "Veenker"
]

# These neighborhoods should be mapped to "Other" by the LLM
RARE_NEIGHBORHOODS = ["Blmngtn", "NPkVill", "Blueste", "Greens", "GrnHill", "Landmrk"]

VALID_QUAL_VALUES = ["Po", "Fa", "TA", "Gd", "Ex"]

ALL_FIELDS = [
    "overall_quality", "gr_liv_area", "garage_cars", "total_basement_sf",
    "full_bath_count", "year_built", "lot_area",
    "neighborhood", "exter_qual", "kitchen_qual"
]

# ── Prompt Version 1: Direct rules ────────────────────────────
PROMPT_V1 = """You are a real estate data extraction assistant for Ames, Iowa.
Extract house features from the user's description. Return ONLY a JSON object.

Fields to extract:
- overall_quality: int 1-10. Quality scale:
    1-2=ruined/very poor, 3=below average, 4=fair, 5=average,
    6=above average, 7=good, 8=very good, 9=excellent, 10=luxury.
    Infer: "luxury/pristine"→9, "great condition"→7, "average"→5,
    "needs work"→3, "fixer-upper"→2.
- gr_liv_area: float, above-grade living area sqft.
    Rough guide: studio≈600, 1bed≈800, 2bed≈1100, 3bed≈1500,
    4bed≈2100, 5bed≈2700.
- garage_cars: int 0-4. Cars the garage holds.
    "no garage"→0, "1-car"→1, "2-car"→2, "big garage"→2, "3-car"→3.
- total_basement_sf: float, basement sqft.
    "no basement"→0. If not mentioned and house style suggests one→null.
- full_bath_count: int. Full bathrooms above grade.
    If not stated: 1bed→1, 2-3bed→2, 4+bed→3.
- year_built: int. Decade hints: "new/modern"→2015, "built 2000s"→2005,
    "90s"→1995, "classic/older"→1970, "vintage"→1945.
- lot_area: float, lot sqft.
    "small"≈5000, "typical"≈8500, "large"≈14000, "estate"≈25000.
- neighborhood: MUST be one of: {neighborhoods}
    "premium/luxury"→"NridgHt", "stone brook"→"StoneBr",
    "nice/good area"→"CollgCr", "average suburb"→"NAmes",
    "affordable"→"OldTown", "college area"→"CollgCr".
    If area is unclear, very small, or not listed → use "Other".
- exter_qual: MUST be one of: {qual_values}
    "luxury exterior"→"Ex", "nice/good"→"Gd", "typical"→"TA",
    "dated/worn"→"Fa", "damaged"→"Po".
- kitchen_qual: MUST be one of: {qual_values}
    "chef kitchen/high-end"→"Ex", "renovated/updated"→"Gd",
    "standard"→"TA", "older/dated"→"Fa".

Rules:
1. Set null for any field you are NOT confident about. Never guess random numbers.
2. Return ONLY the JSON object — no markdown, no extra text.
3. Add a "notes" string for any assumptions you made.

Return:
{{
  "overall_quality": <int or null>,
  "gr_liv_area": <float or null>,
  "garage_cars": <int or null>,
  "total_basement_sf": <float or null>,
  "full_bath_count": <int or null>,
  "year_built": <int or null>,
  "lot_area": <float or null>,
  "neighborhood": <string or null>,
  "exter_qual": <string or null>,
  "kitchen_qual": <string or null>,
  "notes": "<assumptions>"
}}"""

# ── Prompt Version 2: Chain-of-thought ────────────────────────
PROMPT_V2 = """You are an expert real estate analyst in Ames, Iowa.
A user described a property. Extract 10 structured features using step-by-step reasoning.

Think through each feature:
1. What did the user explicitly say?
2. What can be reasonably inferred from context and Ames housing norms?
3. What is genuinely unknown → set null (not a default value).

Feature guide:
- overall_quality (1-10): Physical condition and finish quality.
  Clues: marble/quartz/hardwood→8-9, new construction→8,
  well-maintained→6-7, typical→5, needs updating→4, fixer→2-3.
- gr_liv_area (sqft): Above-ground living space. Cross-check with bedrooms.
  3 bedrooms typically means 1,300-1,800 sqft in Ames.
- garage_cars (0-4): Explicit count is best. "big garage"→2, "oversized"→3.
  No mention of garage + context suggests suburban → could be 1 or 2.
- total_basement_sf (sqft): 0 if "no basement". Unknown = null (not 0!).
  Most Ames homes have a basement.
- full_bath_count: Upstairs + main floor full baths. Half-baths are NOT counted.
- year_built: Use decade clues: "original fixtures"→1960s, "smart home"→2015+,
  "recently built"→2018, "historic"→1930.
- lot_area (sqft): Corner lots are larger. "small lot"≈5000, "normal"≈8500.
- neighborhood: Choose ONLY from: {neighborhoods}
  "great schools district"→"NridgHt" or "StoneBr",
  "near Iowa State"→"CollgCr", "downtown"→"OldTown".
- exter_qual: Choose ONLY from: {qual_values} (Po=poor, Ex=excellent).
  Brick/stone/fiber cement in good condition→"Gd" or "Ex".
- kitchen_qual: Choose ONLY from: {qual_values}.
  Granite+stainless→"Gd" or "Ex", laminate/dated→"TA" or "Fa".

Output ONLY valid JSON, no markdown:
{{
  "overall_quality": <int or null>,
  "gr_liv_area": <float or null>,
  "garage_cars": <int or null>,
  "total_basement_sf": <float or null>,
  "full_bath_count": <int or null>,
  "year_built": <int or null>,
  "lot_area": <float or null>,
  "neighborhood": <string or null>,
  "exter_qual": <string or null>,
  "kitchen_qual": <string or null>,
  "notes": "<reasoning>"
}}"""


def extract_features(user_query: str, prompt_version: int = 1) -> ExtractionResult:
    """
    Stage 1 of the LLM prompt chain.
    Parses a natural language property description into typed feature values.

    Args:
        user_query:     The user's plain-English property description
        prompt_version: 1 (direct rules) or 2 (chain-of-thought)

    Returns:
        ExtractionResult with features, completeness info, and notes.
        Returns an empty ExtractionResult on parse failures — never raises
        for JSON/validation errors (lets the route handler keep running).
    """
    client = OpenAI(api_key=settings.openai_api_key)

    base_prompt = PROMPT_V1 if prompt_version == 1 else PROMPT_V2
    system_prompt = base_prompt.format(
        neighborhoods=", ".join(VALID_NEIGHBORHOODS),
        qual_values=", ".join(VALID_QUAL_VALUES),
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Property description: {user_query}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,   # low = consistent, deterministic extraction
            max_tokens=500,
            timeout=15,  # 15 second timeout to prevent hanging
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        features = HouseFeatures(
            overall_quality   = _safe_int(data.get("overall_quality")),
            gr_liv_area       = _safe_float(data.get("gr_liv_area")),
            garage_cars       = _safe_int(data.get("garage_cars")),
            total_basement_sf = _safe_float(data.get("total_basement_sf")),
            full_bath_count   = _safe_int(data.get("full_bath_count")),
            year_built        = _safe_int(data.get("year_built")),
            lot_area          = _safe_float(data.get("lot_area")),
            neighborhood      = _validate_category(data.get("neighborhood"), VALID_NEIGHBORHOODS),
            exter_qual        = _validate_category(data.get("exter_qual"), VALID_QUAL_VALUES),
            kitchen_qual      = _validate_category(data.get("kitchen_qual"), VALID_QUAL_VALUES),
        )

        extracted = [f for f in ALL_FIELDS if getattr(features, f) is not None]
        missing   = [f for f in ALL_FIELDS if getattr(features, f) is None]
        score     = round(len(extracted) / len(ALL_FIELDS), 2)

        logger.info(
            f"Stage 1 v{prompt_version}: {score:.0%} complete "
            f"({len(extracted)}/10) | extracted={extracted} | missing={missing}"
        )

        return ExtractionResult(
            features=features,
            extracted_fields=extracted,
            missing_fields=missing,
            completeness_score=score,
            notes=data.get("notes", ""),
            prompt_version=prompt_version,
        )

    except json.JSONDecodeError as e:
        logger.error(f"Stage 1: JSON parse error: {e}")
        return _empty_extraction(prompt_version, f"JSON parse error: {e}")

    except (APIError, RateLimitError):
        raise   # let the route handler return a proper HTTP 503

    except Exception as e:
        logger.error(f"Stage 1: unexpected error: {e}", exc_info=True)
        return _empty_extraction(prompt_version, f"Extraction failed: {e}")


# ── Helpers ───────────────────────────────────────────────────

def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _validate_category(value, valid: list[str]) -> Optional[str]:
    """Returns value only if it is in the allowed list (case-insensitive match)."""
    if value is None:
        return None
    if value in valid:
        return value
    for v in valid:
        if v.lower() == str(value).lower():
            return v
    logger.warning(f"Stage 1: invalid categorical value discarded: {value!r}")
    return None


def _empty_extraction(prompt_version: int, notes: str) -> ExtractionResult:
    return ExtractionResult(
        features=HouseFeatures(),
        extracted_fields=[],
        missing_fields=ALL_FIELDS[:],
        completeness_score=0.0,
        notes=notes,
        prompt_version=prompt_version,
    )