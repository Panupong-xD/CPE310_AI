"""
Model evaluation and explainability module.
Evaluates champion model on holdout test set, computes SHAP values,
and exports all production artifacts for backend inference.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)
import shap

from ml_pipeline.preprocess import (
    build_preprocessor,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    BINARY_FEATURES
)

ALL_TRANSFORMED_COLUMNS = NUMERICAL_FEATURES + BINARY_FEATURES


def compute_sample_shap(explainer, feature_names: List[str], X_transformed: np.ndarray) -> Dict[str, float]:
    """Compute local SHAP contributions for a single sample."""
    shap_values = explainer.shap_values(X_transformed)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class (diabetic)
    elif len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    if sv.ndim == 2:
        sample_sv = sv[0]
    else:
        sample_sv = sv

    contributions = {}
    for feat, val in zip(feature_names, sample_sv):
        contributions[feat] = float(val)
        
    return contributions


def run_evaluation(artifacts_dir: str = "ml_pipeline/artifacts") -> Dict[str, Any]:
    """Train champion model on full training set, evaluate on test set, and export artifacts."""
    splits_path = os.path.join(artifacts_dir, "train_test_splits.joblib")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Missing {splits_path}. Run train.py first.")
        
    splits = joblib.load(splits_path)
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    feature_stats = splits["feature_stats"]
    best_model_name = splits["best_model_name"]
    best_model_template = splits["best_model_template"]
    
    print("=" * 70)
    print(f"  FINAL MODEL EVALUATION & ARTIFACT EXPORT: {best_model_name}")
    print("=" * 70)
    
    # 1. Fit complete Pipeline
    preprocessor = build_preprocessor()
    classifier = best_model_template
    
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # 2. Evaluate on Holdout Test Set (20%)
    y_test_prob = model_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    
    test_roc_auc = float(roc_auc_score(y_test, y_test_prob))
    test_pr_auc = float(average_precision_score(y_test, y_test_prob))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_recall = float(recall_score(y_test, y_test_pred))
    test_precision = float(precision_score(y_test, y_test_pred))
    test_f1 = float(f1_score(y_test, y_test_pred))
    
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_specificity = float(tn / (tn + fp))
    
    print("\nHOLDOUT TEST SET PERFORMANCE (n = 1,131):")
    print("-" * 50)
    print(f"ROC-AUC Score:          {test_roc_auc:.4f}")
    print(f"PR-AUC Score:           {test_pr_auc:.4f}")
    print(f"Sensitivity (Recall):   {test_recall:.4f} ({tp}/{tp+fn} cases detected)")
    print(f"Specificity:            {test_specificity:.4f} ({tn}/{tn+fp} healthy identified)")
    print(f"Precision:              {test_precision:.4f}")
    print(f"F1-Score:               {test_f1:.4f}")
    print(f"Accuracy:               {test_acc:.4f}")
    print("-" * 50)
    
    # 3. Fit SHAP TreeExplainer
    print("\nInitializing SHAP TreeExplainer...")
    X_train_trans = model_pipeline.named_steps["preprocessor"].transform(X_train)
    background_summary = shap.sample(X_train_trans, 100, random_state=42)
    tree_explainer = shap.TreeExplainer(classifier, data=background_summary)
    
    base_val = tree_explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_val = float(base_val)
        
    explainer_artifact = {
        "explainer": tree_explainer,
        "feature_names": ALL_TRANSFORMED_COLUMNS,
        "base_value": base_val
    }
    
    # Compute global feature importance ranking
    sample_shap = tree_explainer.shap_values(X_train_trans[:300])
    if isinstance(sample_shap, list):
        sample_shap = sample_shap[1]
    elif len(sample_shap.shape) == 3 and sample_shap.shape[-1] == 2:
        sample_shap = sample_shap[:, :, 1]
        
    mean_abs_shap = np.mean(np.abs(sample_shap), axis=0)
    feature_importance = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in sorted(zip(ALL_TRANSFORMED_COLUMNS, mean_abs_shap), key=lambda x: x[1], reverse=True)
    ]

    # Population risk distribution percentiles
    all_train_probs = model_pipeline.predict_proba(X_train)[:, 1]
    percentile_bins = [float(np.percentile(all_train_probs, p)) for p in range(0, 101, 5)]
    
    # 4. Construct Metadata
    metadata = {
        "model_name": best_model_name,
        "metrics": {
            "test_roc_auc": round(test_roc_auc, 4),
            "test_pr_auc": round(test_pr_auc, 4),
            "test_recall_sensitivity": round(test_recall, 4),
            "test_specificity": round(test_specificity, 4),
            "test_precision": round(test_precision, 4),
            "test_f1": round(test_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)
            }
        },
        "cv_benchmarks": splits["cv_results"],
        "feature_names": ALL_TRANSFORMED_COLUMNS,
        "raw_feature_columns": FEATURE_COLUMNS,
        "feature_stats": feature_stats,
        "feature_importance": feature_importance,
        "percentile_bins": percentile_bins,
        "risk_thresholds": {
            "low_max": 20.0,
            "moderate_max": 50.0,
            "high_min": 50.0
        },
        "risk_categories": {
            "low": "Low Risk",
            "moderate": "Moderate Risk (Pre-diabetes Alert)",
            "high": "High Risk (Action Recommended)"
        }
    }
    
    # 5. Export Artifacts
    pipeline_path = os.path.join(artifacts_dir, "model_pipeline.joblib")
    explainer_path = os.path.join(artifacts_dir, "shap_explainer.joblib")
    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    
    joblib.dump(model_pipeline, pipeline_path)
    joblib.dump(explainer_artifact, explainer_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\nArtifacts successfully exported:")
    print(f"  1. Model Pipeline:   {pipeline_path}")
    print(f"  2. SHAP Explainer:   {explainer_path}")
    print(f"  3. Metadata JSON:    {metadata_path}")
    print("=" * 70)
    
    return metadata


if __name__ == "__main__":
    run_evaluation()
