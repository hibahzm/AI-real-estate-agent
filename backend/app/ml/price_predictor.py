"""
price_predictor.py — Loads the trained ML pipeline and runs price predictions.

The model is loaded ONCE at application startup (not on every request).
This is critical for performance — loading a sklearn pipeline takes ~200ms.

The saved model file contains the FULL pipeline:
  preprocessing (impute + scale + encode) + trained regressor

So when we call predict(), we just pass a raw feature dict and it handles everything.
"""

import json
import logging
import pandas as pd
import joblib
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level variables — loaded once at startup, reused for every request
_pipeline = None       # the full sklearn pipeline
_training_stats = None  # dict from training_statistics.json


def load_model_and_stats() -> None:
    """
    Loads the trained sklearn pipeline and training statistics from disk.
    Called once in main.py's lifespan handler at application startup.
    Raises FileNotFoundError if the model files don't exist.
    """
    global _pipeline, _training_stats

    model_path = Path(settings.model_file_path)
    stats_path = Path(settings.training_stats_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run the Colab notebook first and place the .pkl file in backend/saved_model/"
        )

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Training stats file not found: {stats_path}\n"
            f"Run the Colab notebook first and place training_statistics.json in backend/saved_model/"
        )

    _pipeline = joblib.load(model_path)
    logger.info(f"✅ ML model loaded from {model_path}")

    with open(stats_path, "r") as f:
        _training_stats = json.load(f)
    logger.info(f"✅ Training statistics loaded — median price: ${_training_stats['median_price']:,}")


def predict_price(features: dict) -> float:
    """
    Runs the ML pipeline on a feature dict and returns the predicted price.

    The features dict keys MUST match the column names used during training.
    These are defined in COLUMN_MAPPING in ames_housing_model_training.py and
    mirror the field names in HouseFeatures (house_features_schema.py).

    Args:
        features: dict with keys:
            overall_quality, gr_liv_area, total_basement_sf, garage_area,
            year_built, lot_area, bedroom_count, full_bath_count,
            neighborhood, house_style

    Returns:
        Predicted sale price in dollars (float).

    Raises:
        RuntimeError if the model hasn't been loaded yet.
        ValueError if features are invalid.
    """
    if _pipeline is None:
        raise RuntimeError(
            "ML model not loaded. load_model_and_stats() must be called at startup."
        )

    # Convert dict to a single-row DataFrame.
    # The pipeline's ColumnTransformer expects the ORIGINAL sklearn column names
    # (OverallQual, GrLivArea, …) not the snake_case API names.
    # If the caller passed a HouseFeatures object's .model_dump() we re-map via to_model_input().
    # If the caller already passed sklearn column names (e.g. from unit tests) we use as-is.
    sklearn_columns = [
        "Overall Qual", "Gr Liv Area", "Garage Cars", "Total Bsmt SF",
        "Full Bath", "Year Built", "Lot Area", "Neighborhood",
        "Exter Qual", "Kitchen Qual"
    ]
    if "Overall Qual" not in features:
        # Remap API snake_case field names → AmesHousing.csv column names (with spaces)
        remap = {
            "overall_quality":   "Overall Qual",
            "gr_liv_area":       "Gr Liv Area",
            "garage_cars":       "Garage Cars",
            "total_basement_sf": "Total Bsmt SF",
            "full_bath_count":   "Full Bath",
            "year_built":        "Year Built",
            "lot_area":          "Lot Area",
            "neighborhood":      "Neighborhood",
            "exter_qual":        "Exter Qual",
            "kitchen_qual":      "Kitchen Qual",
        }
        features = {remap.get(k, k): v for k, v in features.items()}

    df = pd.DataFrame([features])

    # Run the full pipeline (preprocessing + prediction)
    prediction = _pipeline.predict(df)[0]

    # Clip to a reasonable range to avoid wild extrapolation
    prediction = float(max(20_000, min(prediction, 800_000)))

    logger.info(f"Price prediction: ${prediction:,.0f} for features: {features}")
    return prediction


def get_training_stats() -> dict:
    """
    Returns the training statistics dict loaded from training_statistics.json.
    Used by Stage 2 to provide market context.

    Raises:
        RuntimeError if not loaded yet.
    """
    if _training_stats is None:
        raise RuntimeError(
            "Training stats not loaded. load_model_and_stats() must be called at startup."
        )
    return _training_stats


def is_loaded() -> bool:
    """Returns True if both the model and stats have been successfully loaded."""
    return _pipeline is not None and _training_stats is not None