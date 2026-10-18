"""
Exploratory Data Analysis (EDA) module for NHANES Diabetes Risk Assessment.
Generates statistical summaries, class distributions, and correlation insights.
"""

import os
import json
import pandas as pd
import numpy as np
from ml_pipeline.preprocess import get_prepared_data, FEATURE_COLUMNS


def run_eda(output_dir: str = "ml_pipeline/artifacts") -> dict:
    """Run exploratory data analysis on the merged NHANES dataset."""
    os.makedirs(output_dir, exist_ok=True)
    X, y, stats = get_prepared_data()
    
    df = X.copy()
    df["is_diabetic"] = y
    
    total_samples = len(df)
    positive_samples = int(y.sum())
    negative_samples = total_samples - positive_samples
    prevalence = float(positive_samples / total_samples)
    
    print("=" * 60)
    print("        NHANES DIABETES EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Total Adult Cohort Size: {total_samples:,}")
    print(f"Diabetic Cases (Positive): {positive_samples:,} ({prevalence * 100:.2f}%)")
    print(f"Non-Diabetic Cases (Negative): {negative_samples:,} ({(1 - prevalence) * 100:.2f}%)")
    print("-" * 60)
    
    # Feature correlations with target
    correlations = df.corr()["is_diabetic"].drop("is_diabetic").sort_values(ascending=False)
    print("\nFeature Pearson Correlation with Diabetes:")
    for feat, corr in correlations.items():
        print(f"  - {feat:<25}: {corr:+.4f}")
        
    # Group comparisons (Diabetic vs Non-Diabetic)
    group_means = df.groupby("is_diabetic").mean().T
    group_means.columns = ["Non-Diabetic Mean", "Diabetic Mean"]
    group_means["Abs Diff"] = (group_means["Diabetic Mean"] - group_means["Non-Diabetic Mean"]).abs()
    group_means["% Change"] = ((group_means["Diabetic Mean"] - group_means["Non-Diabetic Mean"]) / group_means["Non-Diabetic Mean"] * 100)
    
    print("\nGroup Mean Comparison (Diabetic vs Non-Diabetic):")
    print(group_means.to_string())
    
    # Missingness summary
    missing_summary = (df.isnull().sum() / len(df) * 100).to_dict()
    
    eda_summary = {
        "dataset_size": total_samples,
        "prevalence": round(prevalence, 4),
        "class_distribution": {
            "diabetic": positive_samples,
            "non_diabetic": negative_samples
        },
        "feature_correlations": {k: round(v, 4) for k, v in correlations.to_dict().items()},
        "group_means": group_means.round(3).to_dict(),
        "missing_percentages": {k: round(v, 2) for k, v in missing_summary.items()},
        "feature_stats": stats
    }
    
    summary_path = os.path.join(output_dir, "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"\nEDA summary saved to: {summary_path}")
    print("=" * 60)
    return eda_summary


if __name__ == "__main__":
    run_eda()
