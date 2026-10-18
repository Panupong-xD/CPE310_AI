export type BiologicalSex = 'male' | 'female';
export type RiskTier = 'low' | 'moderate' | 'high';

export interface AssessmentFormData {
  age: number;
  gender: BiologicalSex;
  weight_kg: number;
  height_cm: number;
  waist_cm?: number;
  systolic_bp?: number;
  diastolic_bp?: number;
  daily_sugar_g?: number;
  daily_fiber_g?: number;
  daily_calories_kcal?: number;
  hypertension_history: boolean;
  smoking_history: boolean;
  vigorous_activity: boolean;
  sleep_trouble: boolean;
}

export interface FeatureImpact {
  feature: string;
  raw_feature_name: string;
  user_value: string;
  impact: string;
  impact_value: number;
  direction: 'negative' | 'positive';
  advice: string;
}

export interface PredictionResult {
  risk_score: number;
  risk_category: string;
  risk_tier: RiskTier;
  percentile: number;
  imputed_fields: string[];
  screening_positive: boolean;
  decision_threshold: number;
  shap_base_score: number;
  shap_total_contribution_points: number;
  key_drivers: FeatureImpact[];
  protective_factors: FeatureImpact[];
  bmi: number;
  waist_to_height_ratio: number | null;
}

export interface SimulationAdjustments {
  delta_weight_kg: number;
  delta_waist_cm: number;
  delta_sugar_pct: number;
  delta_fiber_g: number;
  active_exercise: boolean;
  manage_hypertension: boolean;
}

export interface SimulationResult {
  original_risk_score: number;
  new_risk_score: number;
  delta_risk_score: number;
  delta_percentage: number;
  original_category: string;
  new_category: string;
  new_risk_tier: RiskTier;
  summary_takeaway: string;
  simulated_bmi: number;
  simulated_waist_to_height_ratio: number | null;
}

export interface ModelMetadata {
  model_name: string;
  metrics: {
    test_roc_auc: number;
    test_pr_auc: number;
    test_recall_sensitivity: number;
    test_specificity: number;
    test_precision: number;
    test_f1: number;
    test_accuracy: number;
  };
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
}
