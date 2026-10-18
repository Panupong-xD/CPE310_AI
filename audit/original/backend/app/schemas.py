"""
Pydantic schemas for request and response validation.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class AssessmentInput(BaseModel):
    """User assessment inputs from non-invasive screening form."""
    age: float = Field(..., ge=18, le=120, description="Age in years (Adults 18+)")
    gender: Literal["male", "female"] = Field(default="male", description="Biological sex ('male' or 'female')")
    weight_kg: float = Field(..., ge=30, le=250, description="Body weight in kilograms")
    height_cm: float = Field(..., ge=100, le=240, description="Standing height in centimeters")
    waist_cm: Optional[float] = Field(default=None, ge=40, le=180, description="Waist circumference in cm")
    systolic_bp: Optional[float] = Field(default=120.0, ge=70, le=240, description="Systolic blood pressure (mmHg)")
    diastolic_bp: Optional[float] = Field(default=80.0, ge=40, le=140, description="Diastolic blood pressure (mmHg)")
    
    # Dietary metrics (daily)
    daily_sugar_g: Optional[float] = Field(default=None, ge=0, le=500, description="Daily total sugar intake in grams")
    daily_fiber_g: Optional[float] = Field(default=None, ge=0, le=150, description="Daily dietary fiber intake in grams")
    daily_calories_kcal: Optional[float] = Field(default=None, ge=500, le=6000, description="Estimated daily caloric intake")
    
    # Questionnaire & Lifestyle indicators
    hypertension_history: Optional[bool] = Field(default=False, description="Told by doctor of high blood pressure")
    smoking_history: Optional[bool] = Field(default=False, description="Smoked 100+ cigarettes in lifetime")
    vigorous_activity: Optional[bool] = Field(default=False, description="Engages in vigorous physical activity")
    sleep_trouble: Optional[bool] = Field(default=False, description="Experienced chronic sleep troubles")


class FeatureImpact(BaseModel):
    """Local SHAP feature contribution and clinical insight."""
    feature: str = Field(..., description="Human-readable feature name")
    raw_feature_name: str = Field(..., description="Internal feature column key")
    user_value: Any = Field(..., description="User's input or estimated value")
    impact: str = Field(..., description="Formatted percentage contribution, e.g., '+12.4%' or '-6.1%'")
    impact_value: float = Field(..., description="Raw SHAP impact value")
    direction: Literal["negative", "positive"] = Field(..., description="'negative' increases risk, 'positive' protects")
    advice: str = Field(..., description="Actionable personalized advice or context")


class PredictionResponse(BaseModel):
    """Complete prediction output with explainability."""
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Predicted risk probability as percentage (0-100%)")
    risk_category: str = Field(..., description="Categorical risk tier: Low, Moderate, or High Risk")
    risk_tier: Literal["low", "moderate", "high"] = Field(..., description="Normalized tier key for UI styling")
    percentile: float = Field(..., ge=0.0, le=100.0, description="Population risk percentile compared to NHANES cohort")
    key_drivers: List[FeatureImpact] = Field(default_factory=list, description="Top risk-increasing factors")
    protective_factors: List[FeatureImpact] = Field(default_factory=list, description="Top protective/low-risk factors")
    bmi: float = Field(..., description="Calculated Body Mass Index")
    waist_to_height_ratio: float = Field(..., description="Calculated Waist-to-Height Ratio")


class SimulationDelta(BaseModel):
    """User adjustments for lifestyle What-If simulation."""
    delta_weight_kg: float = Field(default=0.0, ge=-40.0, le=20.0, description="Weight change in kg (e.g. -5 for 5kg loss)")
    delta_waist_cm: float = Field(default=0.0, ge=-30.0, le=20.0, description="Waist reduction in cm (e.g. -4)")
    delta_sugar_pct: float = Field(default=0.0, ge=-90.0, le=100.0, description="Sugar intake percentage change (e.g. -40%)")
    delta_fiber_g: float = Field(default=0.0, ge=-20.0, le=40.0, description="Fiber increase in grams (e.g. +8g)")
    active_exercise: Optional[bool] = Field(default=None, description="Switch to regular vigorous physical activity")
    manage_hypertension: Optional[bool] = Field(default=None, description="Successfully manage blood pressure")


class SimulationRequest(BaseModel):
    """Simulation request containing baseline inputs and proposed delta changes."""
    baseline: AssessmentInput
    adjustments: SimulationDelta


class SimulationResponse(BaseModel):
    """Results of the What-If simulation."""
    original_risk_score: float
    new_risk_score: float
    delta_risk_score: float
    delta_percentage: float
    original_category: str
    new_category: str
    new_risk_tier: Literal["low", "moderate", "high"]
    summary_takeaway: str
    simulated_bmi: float
    simulated_waist_to_height_ratio: float
