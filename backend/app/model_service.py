"""Serve the actual calibrated classifier and association-only SHAP/what-if outputs."""
import json
import joblib
import numpy as np
import pandas as pd
from backend.app.config import settings
from backend.app.schemas import AssessmentInput, FeatureImpact, PredictionResponse, SimulationRequest, SimulationResponse
from ml_pipeline.preprocess import FEATURE_COLUMNS, engineer_features
from ml_pipeline.explain import compute_sample_shap

FEATURE_DISPLAY_NAMES = {
    'age':'Age (80+ top-coded)', 'gender_is_female':'Biological Sex', 'bmi':'BMI',
    'waist_cm':'Waist Circumference', 'height_cm':'Height', 'weight_kg':'Weight',
    'waist_to_height_ratio':'Waist-to-Height Ratio', 'systolic_bp':'Mean Systolic BP',
    'diastolic_bp':'Mean Diastolic BP', 'daily_sugar_g':'Total Sugar (one-day recall)',
    'daily_fiber_g':'Dietary Fiber', 'daily_calories_kcal':'Calories',
    'sugar_to_fiber_ratio':'Sugar-to-Fiber Ratio', 'hypertension_flag':'Hypertension History / Elevated BP',
    'no_vigorous_recreation_flag':'No Vigorous Recreational Activity',
    'smoking_history_flag':'Ever Smoked 100+ Cigarettes', 'sleep_trouble_flag':'Ever Told Doctor of Sleep Trouble',
}


class ModelService:
    def __init__(self):
        self.pipeline = joblib.load(settings.pipeline_path)
        artifact = joblib.load(settings.explainer_path)
        self.explainer = artifact['explainer']
        self.feature_names = artifact['feature_names']
        self.base_value = artifact['base_value']
        with open(settings.metadata_path) as f:
            self.metadata = json.load(f)
        if self.metadata.get('schema_version') != 2:
            raise ValueError('Artifacts must be rebuilt with the audited pipeline (schema version 2)')

    def get_metadata(self):
        return self.metadata

    def _input_to_dataframe(self, inp):
        def code(value):
            return np.nan if value is None else (1 if value else 2)
        row = {'RIDAGEYR':inp.age,'RIAGENDR':2 if inp.gender=='female' else 1,
               'BMXWT':inp.weight_kg,'BMXHT':inp.height_cm,'BMXWAIST':inp.waist_cm,
               'DR1TSUGR':inp.daily_sugar_g,'DR1TFIBE':inp.daily_fiber_g,
               'DR1TKCAL':inp.daily_calories_kcal,'DR1DRSTZ':1,
               'BPQ020':code(inp.hypertension_history),'PAQ650':code(inp.vigorous_activity),
               'SMQ020':code(inp.smoking_history),'SLQ050':code(inp.sleep_trouble)}
        for i in (1,2,3):
            row[f'BPXSY{i}'] = inp.systolic_bp if i==1 else np.nan
            row[f'BPXDI{i}'] = inp.diastolic_bp if i==1 else np.nan
        raw = pd.DataFrame([row]).astype(float)
        return engineer_features(raw)[FEATURE_COLUMNS]

    def _determine_category(self, score_pct):
        if score_pct < 20:
            return 'Lower model score','low'
        if score_pct <= 50:
            return 'Intermediate model score','moderate'
        return 'Higher model score','high'

    def _calculate_percentile(self, prob):
        bins = self.metadata['percentile_bins']
        return round(float(np.interp(prob,bins,np.linspace(0,100,len(bins)))),1)

    def predict(self, inp):
        X = self._input_to_dataframe(inp)
        prob = float(self.pipeline.predict_proba(X)[0,1])
        category,tier = self._determine_category(prob*100)
        contributions = compute_sample_shap(self.explainer,self.feature_names,X)
        # Check the explanation against the deployed calibrated output.
        if not np.isclose(self.base_value+sum(contributions.values()),prob,atol=1e-6):
            raise RuntimeError('SHAP contributions do not reconstruct deployed probability')
        increasing, decreasing = [],[]
        for feat,value in contributions.items():
            raw = X.iloc[0][feat]
            if abs(value) <= 1e-8:
                continue
            direction = 'raises' if value>0 else 'lowers'
            factor = FeatureImpact(feature=FEATURE_DISPLAY_NAMES[feat],raw_feature_name=feat,
                user_value='Missing (imputed by fitted pipeline)' if pd.isna(raw) else f'{raw:.2f}',
                impact=f'{value*100:+.2f} pp',impact_value=value,
                direction='negative' if value>0 else 'positive',
                advice=f'This feature {direction} the model score relative to its training background. This is a model association, not a causal effect or clinical recommendation.')
            (increasing if value>0 else decreasing).append(factor)
        increasing.sort(key=lambda v:abs(v.impact_value),reverse=True)
        decreasing.sort(key=lambda v:abs(v.impact_value),reverse=True)
        whtr = X.waist_to_height_ratio.iloc[0]
        return PredictionResponse(risk_score=round(prob*100,1),risk_category=category,risk_tier=tier,
            percentile=self._calculate_percentile(prob),key_drivers=increasing[:3],protective_factors=decreasing[:2],
            bmi=round(float(X.bmi.iloc[0]),1),waist_to_height_ratio=None if pd.isna(whtr) else round(float(whtr),2),
            imputed_fields=X.columns[X.iloc[0].isna()].tolist(),
            screening_positive=prob>=self.metadata['decision_threshold'],
            decision_threshold=round(self.metadata['decision_threshold']*100,3),
            shap_base_score=round(self.base_value*100,6),
            shap_total_contribution_points=round(sum(contributions.values())*100,6))

    def simulate(self, req):
        baseline,adj = req.baseline,req.adjustments
        original = self.predict(baseline)
        d = baseline.model_dump()
        stats = self.metadata['feature_stats']
        def resolved(field):
            return d[field] if d[field] is not None else stats[field]['median']
        def clamp(value, low, high):
            return float(np.clip(value,low,high))
        if adj.delta_weight_kg:
            d['weight_kg']=clamp(baseline.weight_kg+adj.delta_weight_kg,30,250)
        if adj.delta_waist_cm:
            d['waist_cm']=clamp(resolved('waist_cm')+adj.delta_waist_cm,40,180)
        if adj.delta_sugar_pct:
            d['daily_sugar_g']=clamp(resolved('daily_sugar_g')*(1+adj.delta_sugar_pct/100),0,500)
        if adj.delta_fiber_g:
            d['daily_fiber_g']=clamp(resolved('daily_fiber_g')+adj.delta_fiber_g,0,150)
        if adj.active_exercise is not None:
            d['vigorous_activity']=adj.active_exercise
        if adj.manage_hypertension:
            d['systolic_bp']=min(resolved('systolic_bp'),120)
            d['diastolic_bp']=min(resolved('diastolic_bp'),80)
            # A current BP scenario cannot erase past diagnosis.
        simulated=self.predict(AssessmentInput(**d)) if d!=baseline.model_dump() else original
        delta=round(original.risk_score-simulated.risk_score,1)
        relative=round(delta/original.risk_score*100,1) if original.risk_score else 0.
        summary=(f'The model score changes from {original.risk_score}% to {simulated.risk_score}% '
                 f'(baseline minus scenario: {delta:+.1f} percentage points). '
                 'This is an association-based input scenario, not a forecast of treatment benefit. '
                 'Missing adjusted measurements use fit-partition medians; values are bounded to the form limits.')
        return SimulationResponse(original_risk_score=original.risk_score,new_risk_score=simulated.risk_score,
            delta_risk_score=delta,delta_percentage=relative,original_category=original.risk_category,
            new_category=simulated.risk_category,new_risk_tier=simulated.risk_tier,summary_takeaway=summary,
            simulated_bmi=simulated.bmi,simulated_waist_to_height_ratio=simulated.waist_to_height_ratio)


model_service=ModelService()
