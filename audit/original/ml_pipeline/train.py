"""
Model training and benchmarking module for Diabetes Risk Screening.
Performs stratified 5-fold cross-validation across multiple algorithms
and selects the best model.
"""

import os
import json
import copy
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score, precision_score

from ml_pipeline.preprocess import get_prepared_data, build_preprocessor, FEATURE_COLUMNS


def get_candidate_models(scale_pos_weight: float = 4.0) -> Dict[str, Any]:
    """Define candidate algorithms for medical screening."""
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            C=0.5
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Try adding LightGBM / XGBoost if OpenMP runtime is available
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    except Exception:
        pass
        
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
    except Exception:
        pass

    return models


def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[str, Dict[str, Any], Any]:
    """
    Run 5-Fold Stratified Cross-Validation on training cohort across candidate models.
    """
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = float(neg_count / max(1, pos_count))
    
    models = get_candidate_models(scale_pos_weight=scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    print(f"\nBenchmarking {len(models)} candidate models on 5-Fold Stratified CV (Train size: {len(X_train)})...")
    print("=" * 85)
    print(f"{'Model':<22} | {'ROC-AUC':<14} | {'PR-AUC':<14} | {'Sensitivity (Recall)':<20} | {'F1-Score':<10}")
    print("-" * 85)
    
    best_score = -1.0
    best_model_name = None
    best_model_template = None
    
    for name, clf in models.items():
        auc_scores = []
        pr_scores = []
        recall_scores = []
        f1_scores = []
        
        for train_idx, val_idx in cv.split(X_train_trans, y_train):
            X_tr, y_tr = X_train_trans[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train_trans[val_idx], y_train.iloc[val_idx]
            
            clf_fold = copy.deepcopy(clf)
            clf_fold.fit(X_tr, y_tr)
            
            y_prob = clf_fold.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            
            auc_scores.append(roc_auc_score(y_val, y_prob))
            pr_scores.append(average_precision_score(y_val, y_prob))
            recall_scores.append(recall_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred))
            
        mean_auc = float(np.mean(auc_scores))
        std_auc = float(np.std(auc_scores))
        mean_pr = float(np.mean(pr_scores))
        mean_rec = float(np.mean(recall_scores))
        mean_f1 = float(np.mean(f1_scores))
        
        results[name] = {
            "roc_auc_mean": round(mean_auc, 4),
            "roc_auc_std": round(std_auc, 4),
            "pr_auc_mean": round(mean_pr, 4),
            "recall_mean": round(mean_rec, 4),
            "f1_mean": round(mean_f1, 4)
        }
        
        print(f"{name:<22} | {mean_auc:.4f} ± {std_auc:.3f} | {mean_pr:.4f}        | {mean_rec:.4f}               | {mean_f1:.4f}")
        
        if mean_auc > best_score:
            best_score = mean_auc
            best_model_name = name
            best_model_template = clf
            
    print("=" * 85)
    print(f"Selected Champion Model: {best_model_name} (CV ROC-AUC: {best_score:.4f})")
    
    return best_model_name, results, best_model_template


def train_and_export(output_dir: str = "ml_pipeline/artifacts") -> Dict[str, Any]:
    """Complete train workflow."""
    os.makedirs(output_dir, exist_ok=True)
    X, y, feature_stats = get_prepared_data()
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f"\nTotal Dataset: {len(X)} rows")
    print(f"Train set: {len(X_train)} samples ({y_train.sum()} positive, {len(y_train) - y_train.sum()} negative)")
    print(f"Test set:  {len(X_test)} samples ({y_test.sum()} positive, {len(y_test) - y_test.sum()} negative)")
    
    best_name, cv_results, best_clf_template = benchmark_models(X_train, y_train)
    
    # Save train/test split data for evaluate.py
    splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_stats": feature_stats,
        "best_model_name": best_name,
        "cv_results": cv_results,
        "best_model_template": best_clf_template
    }
    
    split_path = os.path.join(output_dir, "train_test_splits.joblib")
    joblib.dump(splits, split_path)
    print(f"\nSaved train/test data and champion candidate to: {split_path}")
    
    return splits


if __name__ == "__main__":
    train_and_export()
