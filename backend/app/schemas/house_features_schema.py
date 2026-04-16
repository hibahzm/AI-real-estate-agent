"""
house_features_schema.py — Data shapes for house properties.

The 10 features here MUST match SELECTED_FEATURES in the training notebook
(ames_housing_model_training.py) exactly.  If a field name changes here it
must also change in the notebook — and you need to retrain the model.

Field name → notebook column name mapping:
  overall_quality   → OverallQual
  gr_liv_area       → GrLivArea
  garage_cars       → GarageCars
  total_basement_sf → TotalBsmtSF
  full_bath_count   → FullBath
  year_built        → YearBuilt
  lot_area          → LotArea
  neighborhood      → Neighborhood
  exter_qual        → ExterQual
  kitchen_qual      → KitchenQual

NOTE: the sklearn pipeline receives a DataFrame with the original
column names (OverallQual, GrLivArea, …) not the snake_case API names.
The price_predictor.py converts the snake_case dict back to
original column names before calling pipeline.predict().
"""

from pydantic import BaseModel, Field
from typing import Optional


class HouseFeatures(BaseModel):
    """
    The 10 features the ML model needs for a price prediction.
    All are Optional because Stage 1 may only extract a subset —
    the user fills in the gaps via the UI before prediction runs.
    """

    overall_quality: Optional[int] = Field(
        None, ge=1, le=10,
        description="Overall material and finish quality (1=Very Poor … 10=Very Excellent)"
    )
    gr_liv_area: Optional[float] = Field(
        None, gt=0,
        description="Above-grade (above-ground) living area in square feet"
    )
    garage_cars: Optional[int] = Field(
        None, ge=0, le=5,
        description="Number of cars the garage can hold (0 if no garage)"
    )
    total_basement_sf: Optional[float] = Field(
        None, ge=0,
        description="Total basement area in square feet (0 if no basement)"
    )
    full_bath_count: Optional[int] = Field(
        None, ge=0, le=6,
        description="Full bathrooms above grade"
    )
    year_built: Optional[int] = Field(
        None, ge=1800, le=2025,
        description="Original construction year"
    )
    lot_area: Optional[float] = Field(
        None, gt=0,
        description="Lot size in square feet"
    )
    neighborhood: Optional[str] = Field(
        None,
        description="Physical location within Ames, Iowa (e.g. 'CollgCr', 'NAmes', 'NridgHt')"
    )
    exter_qual: Optional[str] = Field(
        None,
        description="Exterior material quality: Po / Fa / TA / Gd / Ex"
    )
    kitchen_qual: Optional[str] = Field(
        None,
        description="Kitchen quality: Po / Fa / TA / Gd / Ex"
    )

    def get_missing_fields(self) -> list[str]:
        return [f for f, v in self.model_dump().items() if v is None]

    def is_complete(self) -> bool:
        return len(self.get_missing_fields()) == 0

    def to_model_input(self) -> dict:
        """
        Returns a dict with the EXACT column names from AmesHousing.csv
        (space-separated, e.g. "Overall Qual", "Gr Liv Area").
        This is what the ColumnTransformer expects at predict time —
        it was trained with these column names.
        """
        return {
            "Overall Qual":  self.overall_quality,
            "Gr Liv Area":   self.gr_liv_area,
            "Garage Cars":   self.garage_cars,
            "Total Bsmt SF": self.total_basement_sf,
            "Full Bath":     self.full_bath_count,
            "Year Built":    self.year_built,
            "Lot Area":      self.lot_area,
            "Neighborhood":  self.neighborhood,
            "Exter Qual":    self.exter_qual,
            "Kitchen Qual":  self.kitchen_qual,
        }


class ExtractionResult(BaseModel):
    """
    What Stage 1 (the first LLM call) returns.
    Contains extracted features PLUS completeness metadata shown in the UI.
    """
    features:          HouseFeatures
    extracted_fields:  list[str]   # which of the 10 fields were successfully found
    missing_fields:    list[str]   # which still need the user to fill them in
    completeness_score: float      # 0.0 = nothing extracted, 1.0 = all 10 found
    notes:             str = ""    # LLM assumptions / caveats
    prompt_version:    int = 1     # which A/B prompt variant was used


class PredictionRequest(BaseModel):
    """Request body for POST /api/v1/extract."""
    user_query:     str = Field(..., min_length=5,
                                description="Natural language property description")
    prompt_version: int = Field(1, ge=1, le=2,
                                description="Stage 1 prompt variant: 1 or 2")


class InsightRequest(BaseModel):
    """Request body for POST /api/v1/insights."""
    user_query: str = Field(..., min_length=5,
                            description="A market question")