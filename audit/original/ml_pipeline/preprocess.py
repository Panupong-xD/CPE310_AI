"""
Preprocess module for NHANES Diabetes Risk Screening.
Ingests, merges, cleans, and engineers features from NHANES CSV tables.
"""

import os
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler


def find_dataset_dir(base_dir: str = ".") -> str:
    """Find dataset directory (supporting lowercase 'dataset' or 'Dataset')."""
    candidates = [
        os.path.join(base_dir, "dataset"),
        os.path.join(base_dir, "Dataset"),
        os.path.join(base_dir, "..", "dataset"),
        os.path.join(base_dir, "..", "Dataset"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "demographic.csv")):
            return c
    raise FileNotFoundError("Could not locate NHANES dataset directory.")


def get_diabetic_medication_seqns(dataset_dir: str) -> set:
    """Identify SEQN of participants taking anti-diabetic medications."""
    med_path = os.path.join(dataset_dir, "medications.csv")
    if not os.path.exists(med_path):
        return set()
    
    df_med = pd.read_csv(med_path, encoding_errors="ignore", low_memory=False)
    diab_keywords = [
        "INSULIN", "METFORMIN", "GLIP", "GLYB", "JANUVIA", "PIOGLITAZONE",
        "DIABET", "GLUCO", "SITAGLIPTIN", "REPAGLINIDE", "ACARBOSE",
        "ROSIGLITAZONE", "GLIMEPIRIDE", "EXENATIDE", "LIRAGLUTIDE",
        "EMPAGLIFLOZIN", "DAPAGLIFLOZIN", "CANAGLIFLOZIN", "LINAGLIPTIN"
    ]
    
    # Check RXDDRUG and any string columns for diabetes drug names
    pattern = "|".join(diab_keywords)
    diab_mask = pd.Series(False, index=df_med.index)
    
    for col in ["RXDDRUG", "RXDDRGID", "RXDRSC1", "RXDRSC2", "RXDRSC3", "RXDRSD1", "RXDRSD2", "RXDRSD3"]:
        if col in df_med.columns:
            matches = df_med[col].astype(str).str.upper().str.contains(pattern, na=False)
            diab_mask = diab_mask | matches
            
    diab_seqns = set(df_med.loc[diab_mask, "SEQN"].dropna().unique())
    return diab_seqns


def load_and_merge_data(dataset_dir: str = None) -> pd.DataFrame:
    """Load and merge all NHANES tables on SEQN."""
    if dataset_dir is None:
        dataset_dir = find_dataset_dir()
        
    print(f"Loading NHANES data from: {dataset_dir}")
    
    # 1. Demographic (Base table)
    demo_df = pd.read_csv(os.path.join(dataset_dir, "demographic.csv"), low_memory=False)
    print(f"Demographic shape: {demo_df.shape}")
    
    # Filter adults (RIDAGEYR >= 20)
    demo_df = demo_df[demo_df["RIDAGEYR"] >= 20].copy()
    print(f"Adult cohort (RIDAGEYR >= 20): {len(demo_df)} rows")
    
    # 2. Examination
    exam_cols = ["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BPXSY1", "BPXDI1"]
    exam_df = pd.read_csv(os.path.join(dataset_dir, "examination.csv"), usecols=lambda c: c in exam_cols, low_memory=False)
    
    # 3. Labs
    lab_cols = ["SEQN", "LBXGH", "LBXSGL"]
    lab_df = pd.read_csv(os.path.join(dataset_dir, "labs.csv"), usecols=lambda c: c in lab_cols, low_memory=False)
    
    # 4. Diet
    diet_cols = ["SEQN", "DR1TKCAL", "DR1TSUGR", "DR1TFIBE"]
    diet_df = pd.read_csv(os.path.join(dataset_dir, "diet.csv"), usecols=lambda c: c in diet_cols, low_memory=False)
    
    # 5. Questionnaire
    quest_cols = ["SEQN", "DIQ010", "BPQ020", "SMQ020", "PAQ605", "SLQ050"]
    quest_df = pd.read_csv(os.path.join(dataset_dir, "questionnaire.csv"), usecols=lambda c: c in quest_cols, low_memory=False)
    
    # Merge tables using left join on adult demographics
    merged = demo_df[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
    merged = merged.merge(exam_df, on="SEQN", how="left")
    merged = merged.merge(lab_df, on="SEQN", how="left")
    merged = merged.merge(diet_df, on="SEQN", how="left")
    merged = merged.merge(quest_df, on="SEQN", how="left")
    
    # Medications
    med_diab_seqns = get_diabetic_medication_seqns(dataset_dir)
    merged["on_diabetic_med"] = merged["SEQN"].isin(med_diab_seqns).astype(int)
    
    print(f"Merged dataset shape: {merged.shape}")
    return merged


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define ground truth target: is_diabetic (1 = Diabetic, 0 = Non-diabetic)
    Criteria:
    - 1: LBXGH >= 6.5 OR LBXSGL >= 126.0 OR DIQ010 == 1 OR on_diabetic_med == 1
    - 0: (LBXGH < 5.7 or NaN) and (LBXSGL < 100 or NaN) and DIQ010 == 2 and on_diabetic_med == 0
    Exclude rows with indeterminate target status.
    """
    df = df.copy()
    
    # Condition 1: Positive for diabetes
    cond_positive = (
        (df["LBXGH"] >= 6.5) |
        (df["LBXSGL"] >= 126.0) |
        (df["DIQ010"] == 1) |
        (df["on_diabetic_med"] == 1)
    )
    
    # Condition 0: Negative for diabetes
    # Non-diabetic response (DIQ010 == 2), no elevated lab readings, not on diabetic meds
    cond_negative = (
        (df["DIQ010"] == 2) &
        (df["on_diabetic_med"] == 0) &
        ((df["LBXGH"].isna()) | (df["LBXGH"] < 6.5)) &
        ((df["LBXSGL"].isna()) | (df["LBXSGL"] < 126.0))
    )
    
    df["is_diabetic"] = np.nan
    df.loc[cond_negative, "is_diabetic"] = 0
    df.loc[cond_positive, "is_diabetic"] = 1
    
    # Drop rows without determinable target
    initial_len = len(df)
    df = df.dropna(subset=["is_diabetic"]).copy()
    df["is_diabetic"] = df["is_diabetic"].astype(int)
    
    print(f"Target defined: {len(df)} records kept (dropped {initial_len - len(df)} indeterminate).")
    print(f"Target class distribution:\n{df['is_diabetic'].value_counts(normalize=True).round(4) * 100}%")
    print(f"Counts: \n{df['is_diabetic'].value_counts()}")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute non-invasive engineered features for Web UI inference.
    """
    df = df.copy()
    
    # Demographics
    df["age"] = df["RIDAGEYR"]
    df["gender_is_female"] = (df["RIAGENDR"] == 2).astype(int)
    
    # Anthropometrics
    df["weight_kg"] = df["BMXWT"]
    df["height_cm"] = df["BMXHT"]
    df["bmi"] = df["BMXBMI"]
    df["waist_cm"] = df["BMXWAIST"]
    
    # If BMI is missing but height and weight exist, compute it: BMI = kg / (m^2)
    bmi_missing = df["bmi"].isna() & df["weight_kg"].notna() & df["height_cm"].notna()
    df.loc[bmi_missing, "bmi"] = df.loc[bmi_missing, "weight_kg"] / ((df.loc[bmi_missing, "height_cm"] / 100) ** 2)
    
    # Feature 1: Waist to Height Ratio
    df["waist_to_height_ratio"] = df["waist_cm"] / (df["height_cm"].replace(0, np.nan))
    
    # Blood Pressure
    df["systolic_bp"] = df["BPXSY1"]
    df["diastolic_bp"] = df["BPXDI1"]
    
    # Dietary metrics
    df["daily_sugar_g"] = df["DR1TSUGR"]
    df["daily_fiber_g"] = df["DR1TFIBE"]
    df["daily_calories_kcal"] = df["DR1TKCAL"]
    
    # Feature 2: Sugar to Fiber Ratio
    df["sugar_to_fiber_ratio"] = df["daily_sugar_g"] / (df["daily_fiber_g"] + 1e-5)
    
    # Lifestyle Indicators (1 = Risk factor present, 0 = Absent)
    # Hypertension history: BPQ020 == 1 or Systolic >= 140 or Diastolic >= 90
    df["hypertension_flag"] = (
        (df["BPQ020"] == 1) | (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)
    ).astype(float)
    df.loc[df["BPQ020"].isna() & df["systolic_bp"].isna() & df["diastolic_bp"].isna(), "hypertension_flag"] = np.nan
    
    # Physical activity: PAQ605 (1 = Yes vigorous, 2 = No) -> Sedentary flag (1 = Sedentary, 0 = Active)
    df["sedentary_flag"] = df["PAQ605"].map({1: 0.0, 2: 1.0})
    
    # Smoking history: SMQ020 (1 = Yes >=100 cigarettes in life, 2 = No)
    df["smoking_history_flag"] = df["SMQ020"].map({1: 1.0, 2: 0.0})
    
    # Sleep trouble: SLQ050 (1 = Yes told doctor sleep trouble, 2 = No)
    df["sleep_trouble_flag"] = df["SLQ050"].map({1: 1.0, 2: 0.0})
    
    return df


FEATURE_COLUMNS = [
    "age",
    "gender_is_female",
    "bmi",
    "waist_cm",
    "height_cm",
    "weight_kg",
    "waist_to_height_ratio",
    "systolic_bp",
    "diastolic_bp",
    "daily_sugar_g",
    "daily_fiber_g",
    "daily_calories_kcal",
    "sugar_to_fiber_ratio",
    "hypertension_flag",
    "sedentary_flag",
    "smoking_history_flag",
    "sleep_trouble_flag"
]

NUMERICAL_FEATURES = [
    "age", "bmi", "waist_cm", "height_cm", "weight_kg",
    "waist_to_height_ratio", "systolic_bp", "diastolic_bp",
    "daily_sugar_g", "daily_fiber_g", "daily_calories_kcal",
    "sugar_to_fiber_ratio"
]

BINARY_FEATURES = [
    "gender_is_female", "hypertension_flag", "sedentary_flag",
    "smoking_history_flag", "sleep_trouble_flag"
]


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer with median imputation and RobustScaler."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
    
    bin_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_FEATURES),
        ("bin", bin_pipeline, BINARY_FEATURES)
    ], remainder="drop")
    
    return preprocessor


def get_prepared_data(dataset_dir: str = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Complete preparation workflow returning X, y, and metadata defaults."""
    merged = load_and_merge_data(dataset_dir)
    labeled = define_target(merged)
    engineered = engineer_features(labeled)
    
    X = engineered[FEATURE_COLUMNS].copy()
    y = engineered["is_diabetic"].copy()
    
    # Calculate population medians and stats for missing UI input fallback
    stats = {}
    for col in FEATURE_COLUMNS:
        stats[col] = {
            "median": float(X[col].median()),
            "mean": float(X[col].mean()),
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "p25": float(X[col].quantile(0.25)),
            "p75": float(X[col].quantile(0.75)),
        }
        
    return X, y, stats


if __name__ == "__main__":
    X, y, stats = get_prepared_data()
    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(f"Missing values in features:\n{X.isnull().sum()}")
