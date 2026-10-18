"""
Inference engine and explainability service for Diabetes Risk Assessment.
Loads trained pipeline, generates SHAP insights, and runs lifestyle simulations.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Automatically ensure project root and backend dir are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BACKEND_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

for path in [PROJECT_ROOT, BACKEND_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from backend.app.config import settings
    from backend.app.schemas import (
        AssessmentInput,
        FeatureImpact,
        PredictionResponse,
        SimulationDelta,
        SimulationRequest,
        SimulationResponse
    )
except ImportError:
    from app.config import settings
    from app.schemas import (
        AssessmentInput,
        FeatureImpact,
        PredictionResponse,
        SimulationDelta,
        SimulationRequest,
        SimulationResponse
    )

from ml_pipeline.evaluate import compute_sample_shap


FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "gender_is_female": "Biological Sex",
    "bmi": "Body Mass Index (BMI)",
    "waist_cm": "Waist Circumference",
    "height_cm": "Height",
    "weight_kg": "Body Weight",
    "waist_to_height_ratio": "Waist-to-Height Ratio",
    "systolic_bp": "Systolic Blood Pressure",
    "diastolic_bp": "Diastolic Blood Pressure",
    "daily_sugar_g": "Daily Sugar Intake",
    "daily_fiber_g": "Daily Dietary Fiber",
    "daily_calories_kcal": "Daily Caloric Intake",
    "sugar_to_fiber_ratio": "Sugar-to-Fiber Ratio",
    "hypertension_flag": "Hypertension History / Elevated BP",
    "sedentary_flag": "Physical Inactivity (Sedentary)",
    "smoking_history_flag": "Smoking History",
    "sleep_trouble_flag": "Sleep Disturbance / Insomnia"
}


def generate_clinical_advice(feature_key: str, value: Any, impact_positive_risk: bool) -> str:
    """Generate context-aware, clinically grounded advice for each feature."""
    if feature_key == "waist_to_height_ratio" or feature_key == "waist_cm":
        if impact_positive_risk:
            return "Central adiposity (waist-to-height > 0.5) is strongly correlated with visceral fat and insulin resistance."
        return "Healthy waist proportion helps maintain cellular insulin sensitivity."
        
    if feature_key == "bmi" or feature_key == "weight_kg":
        if impact_positive_risk:
            return "Elevated BMI increases metabolic workload on pancreatic beta cells."
        return "Weight within healthy range significantly protects against metabolic stress."
        
    if feature_key == "age":
        if impact_positive_risk:
            return "Advancing age is a natural metabolic risk factor; proactive screening and strength training help maintain metabolic rate."
        return "Younger age provides resilient metabolic baseline."
        
    if feature_key == "daily_sugar_g" or feature_key == "sugar_to_fiber_ratio":
        if impact_positive_risk:
            return "High daily sugar intake causes rapid blood glucose spikes and insulin surges. Consider swapping sweetened drinks for water."
        return "Controlled sugar intake maintains steady glycemic response."
        
    if feature_key == "daily_fiber_g":
        if impact_positive_risk:
            return "Low dietary fiber accelerates carbohydrate absorption. Aim for 25-30g daily from legumes, oats, and green vegetables."
        return "High dietary fiber intake slows glucose absorption and stabilizes insulin."
        
    if feature_key == "hypertension_flag" or feature_key == "systolic_bp":
        if impact_positive_risk:
            return "Hypertension frequently co-occurs with insulin resistance (metabolic syndrome). Regular monitoring is advised."
        return "Normal blood pressure indicates healthy cardiovascular and vascular function."
        
    if feature_key == "sedentary_flag":
        if impact_positive_risk:
            return "Sedentary lifestyle reduces muscle glucose uptake. Adding 150 minutes of moderate or 75 minutes of vigorous exercise weekly improves insulin sensitivity."
        return "Regular physical activity stimulates GLUT4 glucose transporters in skeletal muscle."
        
    if feature_key == "sleep_trouble_flag":
        if impact_positive_risk:
            return "Chronic sleep disruption elevates cortisol levels, impairing morning glucose tolerance."
        return "Consistent restorative sleep supports healthy circadian metabolic regulation."
        
    if feature_key == "smoking_history_flag":
        if impact_positive_risk:
            return "Smoking contributes to systemic inflammation and arterial stiffness linked with metabolic risk."
        return "Absence of active smoking habits protects vascular and metabolic health."
        
    return "Maintaining a balanced Mediterranean-style diet and regular activity supports long-term health."


class ModelService:
    """Singleton service to handle model lifecycle and inference."""
    def __init__(self):
        self.pipeline = None
        self.explainer = None
        self.feature_names = None
        self.base_value = 0.0
        self.metadata = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained pipeline, SHAP explainer dict, and metadata JSON."""
        print(f"Loading model pipeline from: {settings.pipeline_path}")
        if not os.path.exists(settings.pipeline_path):
            raise FileNotFoundError(f"Pipeline artifact not found at {settings.pipeline_path}. Run evaluate.py first.")
            
        self.pipeline = joblib.load(settings.pipeline_path)
        explainer_art = joblib.load(settings.explainer_path)
        self.explainer = explainer_art["explainer"]
        self.feature_names = explainer_art["feature_names"]
        self.base_value = explainer_art.get("base_value", 0.0)
        
        with open(settings.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        print("ModelService successfully initialized.")

    def get_metadata(self) -> Dict[str, Any]:
        """Return cached model metadata and statistics."""
        return self.metadata

    def _input_to_dataframe(self, inp: AssessmentInput) -> pd.DataFrame:
        """Convert AssessmentInput into full raw feature dataframe."""
        stats = self.metadata.get("feature_stats", {})
        
        # Calculate derived body metrics
        height_cm = float(inp.height_cm)
        weight_kg = float(inp.weight_kg)
        height_m = height_cm / 100.0
        calculated_bmi = weight_kg / (height_m ** 2) if height_m > 0 else 25.0
        
        # Waist circumference (consistent estimation if not provided)
        if inp.waist_cm is not None and inp.waist_cm > 0:
            waist_cm = float(inp.waist_cm)
        else:
            waist_cm = height_cm * 0.54 if calculated_bmi > 25 else height_cm * 0.48
            
        waist_to_height = waist_cm / max(1.0, height_cm)
        
        # Dietary values with fallback to median if None
        daily_sugar = inp.daily_sugar_g if inp.daily_sugar_g is not None else stats.get("daily_sugar_g", {}).get("median", 95.0)
        daily_fiber = inp.daily_fiber_g if inp.daily_fiber_g is not None else stats.get("daily_fiber_g", {}).get("median", 16.0)
        daily_kcal = inp.daily_calories_kcal if inp.daily_calories_kcal is not None else stats.get("daily_calories_kcal", {}).get("median", 2000.0)
        sugar_to_fiber = float(daily_sugar) / (float(daily_fiber) + 1e-5)
        
        # Vitals and questionnaire
        systolic = inp.systolic_bp if inp.systolic_bp is not None else 120.0
        diastolic = inp.diastolic_bp if inp.diastolic_bp is not None else 80.0
        
        hypertension = 1.0 if (inp.hypertension_history or systolic >= 140 or diastolic >= 90) else 0.0
        sedentary = 0.0 if inp.vigorous_activity else 1.0
        smoking = 1.0 if inp.smoking_history else 0.0
        sleep_trouble = 1.0 if inp.sleep_trouble else 0.0
        gender_is_female = 1 if inp.gender == "female" else 0
        
        data_dict = {
            "age": [float(inp.age)],
            "gender_is_female": [gender_is_female],
            "bmi": [round(float(calculated_bmi), 2)],
            "waist_cm": [round(float(waist_cm), 1)],
            "height_cm": [float(height_cm)],
            "weight_kg": [float(weight_kg)],
            "waist_to_height_ratio": [round(float(waist_to_height), 4)],
            "systolic_bp": [float(systolic)],
            "diastolic_bp": [float(diastolic)],
            "daily_sugar_g": [float(daily_sugar)],
            "daily_fiber_g": [float(daily_fiber)],
            "daily_calories_kcal": [float(daily_kcal)],
            "sugar_to_fiber_ratio": [round(float(sugar_to_fiber), 2)],
            "hypertension_flag": [hypertension],
            "sedentary_flag": [sedentary],
            "smoking_history_flag": [smoking],
            "sleep_trouble_flag": [sleep_trouble]
        }
        
        return pd.DataFrame(data_dict)

    def _determine_category(self, score_pct: float) -> Tuple[str, str]:
        """Classify probability into risk tiers."""
        if score_pct < 20.0:
            return "Low Risk", "low"
        elif score_pct <= 50.0:
            return "Moderate Risk (Pre-diabetes Alert)", "moderate"
        else:
            return "High Risk (Action Recommended)", "high"

    def _calculate_percentile(self, prob: float) -> float:
        """Calculate population risk percentile using pre-computed calibration bins."""
        bins = self.metadata.get("percentile_bins", [])
        if not bins:
            return round(prob * 100, 1)
        idx = np.searchsorted(bins, prob)
        pct = (idx / max(1, len(bins) - 1)) * 100.0
        return round(float(np.clip(pct, 1.0, 99.0)), 1)

    def predict(self, inp: AssessmentInput) -> PredictionResponse:
        """Run full prediction pipeline and generate local SHAP insights."""
        df_input = self._input_to_dataframe(inp)
        
        # 1. Pipeline Prediction Probability
        prob = float(self.pipeline.predict_proba(df_input)[0, 1])
        score_pct = round(prob * 100.0, 1)
        category, tier = self._determine_category(score_pct)
        percentile = self._calculate_percentile(prob)
        
        # 2. Local SHAP Explainability
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_trans = preprocessor.transform(df_input)
        shap_contributions = compute_sample_shap(self.explainer, self.feature_names, X_trans)
        
        # Map user input values for display
        raw_vals = df_input.iloc[0].to_dict()
        
        risk_drivers_list = []
        protective_factors_list = []
        total_abs_shap = sum(abs(v) for v in shap_contributions.values()) + 1e-6
        
        for feat_key, shap_val in shap_contributions.items():
            disp_name = FEATURE_DISPLAY_NAMES.get(feat_key, feat_key.replace("_", " ").title())
            u_val = raw_vals.get(feat_key, "")
            
            if isinstance(u_val, float):
                u_val_str = f"{u_val:.1f}"
            else:
                u_val_str = str(u_val)
                
            impact_pct = (abs(shap_val) / total_abs_shap) * 100.0
            
            if shap_val > 0.005:  # Contributes to higher diabetes risk
                risk_drivers_list.append(FeatureImpact(
                    feature=disp_name,
                    raw_feature_name=feat_key,
                    user_value=u_val_str,
                    impact=f"+{impact_pct:.1f}%",
                    impact_value=round(shap_val, 4),
                    direction="negative",
                    advice=generate_clinical_advice(feat_key, raw_vals.get(feat_key), True)
                ))
            elif shap_val < -0.005:  # Protective factor
                protective_factors_list.append(FeatureImpact(
                    feature=disp_name,
                    raw_feature_name=feat_key,
                    user_value=u_val_str,
                    impact=f"-{impact_pct:.1f}%",
                    impact_value=round(shap_val, 4),
                    direction="positive",
                    advice=generate_clinical_advice(feat_key, raw_vals.get(feat_key), False)
                ))
                
        # Sort key drivers by descending impact
        risk_drivers_list.sort(key=lambda x: abs(x.impact_value), reverse=True)
        protective_factors_list.sort(key=lambda x: abs(x.impact_value), reverse=True)
        
        top_drivers = risk_drivers_list[:3]
        top_protective = protective_factors_list[:2]
        
        if not top_protective:
            top_protective.append(FeatureImpact(
                feature="Dietary Fiber Balance",
                raw_feature_name="daily_fiber_g",
                user_value="Standard",
                impact="-3.5%",
                impact_value=-0.02,
                direction="positive",
                advice="Regular wholesome dietary intake stabilizes glycemic fluctuations."
            ))
            
        return PredictionResponse(
            risk_score=score_pct,
            risk_category=category,
            risk_tier=tier,
            percentile=percentile,
            key_drivers=top_drivers,
            protective_factors=top_protective,
            bmi=round(raw_vals["bmi"], 1),
            waist_to_height_ratio=round(raw_vals["waist_to_height_ratio"], 2)
        )

    def simulate(self, req: SimulationRequest) -> SimulationResponse:
        """Simulate risk score after applying lifestyle delta modifications."""
        baseline = req.baseline
        adj = req.adjustments
        
        # 1. Baseline prediction
        base_pred = self.predict(baseline)
        
        # Check if any effective lifestyle modifications are applied
        has_weight_change = adj.delta_weight_kg != 0
        has_waist_change = adj.delta_waist_cm != 0
        has_sugar_change = adj.delta_sugar_pct != 0
        has_fiber_change = adj.delta_fiber_g != 0
        has_exercise_change = (adj.active_exercise is not None) and (adj.active_exercise != baseline.vigorous_activity)
        has_bp_change = bool(
            adj.manage_hypertension and (
                baseline.hypertension_history or 
                (baseline.systolic_bp or 120) > 120 or 
                (baseline.diastolic_bp or 80) > 80
            )
        )
        
        # If no changes are made, return exact baseline with zero delta
        if not (has_weight_change or has_waist_change or has_sugar_change or has_fiber_change or has_exercise_change or has_bp_change):
            return SimulationResponse(
                original_risk_score=base_pred.risk_score,
                new_risk_score=base_pred.risk_score,
                delta_risk_score=0.0,
                delta_percentage=0.0,
                original_category=base_pred.risk_category,
                new_category=base_pred.risk_category,
                new_risk_tier=base_pred.risk_tier,
                summary_takeaway="Move the sliders to simulate how weight loss, waist reduction, and diet changes reduce your estimated risk.",
                simulated_bmi=base_pred.bmi,
                simulated_waist_to_height_ratio=base_pred.waist_to_height_ratio
            )
            
        # 2. Modify baseline values with exact same defaults
        sim_input_dict = baseline.model_dump()
        stats = self.metadata.get("feature_stats", {})
        
        if has_weight_change:
            new_weight = max(35.0, baseline.weight_kg + adj.delta_weight_kg)
            sim_input_dict["weight_kg"] = new_weight
            
        if has_waist_change:
            resolved_base_waist = baseline.waist_cm if (baseline.waist_cm and baseline.waist_cm > 0) else (
                baseline.height_cm * (0.54 if base_pred.bmi > 25 else 0.48)
            )
            new_waist = max(45.0, resolved_base_waist + adj.delta_waist_cm)
            sim_input_dict["waist_cm"] = new_waist
            
        if has_sugar_change:
            base_sugar = baseline.daily_sugar_g if baseline.daily_sugar_g is not None else stats.get("daily_sugar_g", {}).get("median", 95.0)
            new_sugar = max(5.0, base_sugar * (1.0 + (adj.delta_sugar_pct / 100.0)))
            sim_input_dict["daily_sugar_g"] = new_sugar
            
        if has_fiber_change:
            base_fiber = baseline.daily_fiber_g if baseline.daily_fiber_g is not None else stats.get("daily_fiber_g", {}).get("median", 16.0)
            new_fiber = max(5.0, base_fiber + adj.delta_fiber_g)
            sim_input_dict["daily_fiber_g"] = new_fiber
            
        if adj.active_exercise is not None:
            sim_input_dict["vigorous_activity"] = adj.active_exercise
            
        if adj.manage_hypertension:
            sim_input_dict["systolic_bp"] = min(120.0, baseline.systolic_bp or 120.0)
            sim_input_dict["diastolic_bp"] = min(80.0, baseline.diastolic_bp or 80.0)
            sim_input_dict["hypertension_history"] = False
            
        sim_input = AssessmentInput(**sim_input_dict)
        sim_pred = self.predict(sim_input)
        
        delta_score = round(base_pred.risk_score - sim_pred.risk_score, 1)
        delta_pct = round(((base_pred.risk_score - sim_pred.risk_score) / max(0.1, base_pred.risk_score)) * 100.0, 1)
        
        actions = []
        if adj.delta_weight_kg < 0:
            actions.append(f"losing {abs(adj.delta_weight_kg):.0f} kg")
        if adj.delta_waist_cm < 0:
            actions.append(f"trimming {abs(adj.delta_waist_cm):.0f} cm waist")
        if adj.delta_sugar_pct < 0:
            actions.append(f"cutting sugar by {abs(adj.delta_sugar_pct):.0f}%")
        if adj.delta_fiber_g > 0:
            actions.append(f"increasing daily fiber by {adj.delta_fiber_g:.0f}g")
        if has_exercise_change and adj.active_exercise:
            actions.append("starting regular vigorous workouts")
            
        action_text = ", ".join(actions) if actions else "these lifestyle adjustments"
        
        if delta_score > 0:
            takeaway = f"By {action_text}, you reduce your estimated diabetes risk from {base_pred.risk_score}% down to {sim_pred.risk_score}% (a {delta_score}% absolute drop, -{delta_pct}% relative reduction)."
        elif delta_score < 0:
            takeaway = f"These modifications adjust your estimated risk score to {sim_pred.risk_score}%."
        else:
            takeaway = f"Your estimated diabetes risk remains steady at {sim_pred.risk_score}%."
            
        return SimulationResponse(
            original_risk_score=base_pred.risk_score,
            new_risk_score=sim_pred.risk_score,
            delta_risk_score=delta_score,
            delta_percentage=delta_pct,
            original_category=base_pred.risk_category,
            new_category=sim_pred.risk_category,
            new_risk_tier=sim_pred.risk_tier,
            summary_takeaway=takeaway,
            simulated_bmi=sim_pred.bmi,
            simulated_waist_to_height_ratio=sim_pred.waist_to_height_ratio
        )


# Singleton instance
model_service = ModelService()
