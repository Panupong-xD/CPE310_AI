"""Deterministic NHANES 2013–2014 cohort, labels and shared feature engineering.
No fitted statistics are learned here; imputation/scaling belong inside each fold.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'ml_pipeline' / 'artifacts'
NUMERICAL_FEATURES = [
    'age', 'bmi', 'waist_cm', 'height_cm', 'weight_kg', 'waist_to_height_ratio',
    'systolic_bp', 'diastolic_bp', 'daily_sugar_g', 'daily_fiber_g',
    'daily_calories_kcal', 'sugar_to_fiber_ratio',
]
BINARY_FEATURES = ['gender_is_female', 'hypertension_flag', 'no_vigorous_recreation_flag',
                   'smoking_history_flag', 'sleep_trouble_flag']
FEATURE_COLUMNS = NUMERICAL_FEATURES + BINARY_FEATURES
# Match whole ingredient tokens, never indications or vague GLIP/GLUCO substrings.
DRUG_PATTERN = (r'\b(?:INSULIN(?: [A-Z]+)?|METFORMIN|GLIPIZIDE|GLYBURIDE|GLIBENCLAMIDE|'
                r'GLIMEPIRIDE|PIOGLITAZONE|ROSIGLITAZONE|SITAGLIPTIN|SAXAGLIPTIN|'
                r'LINAGLIPTIN|ALOGLIPTIN|REPAGLINIDE|NATEGLINIDE|ACARBOSE|MIGLITOL|'
                r'EXENATIDE|LIRAGLUTIDE|CANAGLIFLOZIN|DAPAGLIFLOZIN|EMPAGLIFLOZIN|'
                r'PRAMLINTIDE|TOLBUTAMIDE|TOLAZAMIDE|CHLORPROPAMIDE)\b')


def find_dataset_dir(base_dir=None):
    base = Path(base_dir) if base_dir else ROOT
    for path in (base / 'Dataset', base / 'dataset', base.parent / 'Dataset'):
        if (path / 'demographic.csv').exists():
            return str(path)
    raise FileNotFoundError('Cannot locate Dataset/demographic.csv')


def read_table(dataset_dir, name, columns=None, unique=True):
    path = Path(dataset_dir) / f'{name}.csv'
    # Medication names contain Latin-1 characters; decoding errors must not be discarded.
    df = pd.read_csv(path, encoding='latin1', low_memory=False,
                     usecols=(lambda c: c in columns) if columns else None)
    if columns and (missing := set(columns) - set(df.columns)):
        raise ValueError(f'{name}: missing required columns {sorted(missing)}')
    if df.SEQN.isna().any() or (unique and df.SEQN.duplicated().any()):
        raise ValueError(f'{name}: SEQN must be non-null and unique')
    return df


def get_diabetic_medication_seqns(dataset_dir):
    med = read_table(dataset_dir, 'medications', ['SEQN', 'RXDUSE', 'RXDDRUG'], unique=False)
    mask = med.RXDUSE.eq(1) & med.RXDDRUG.fillna('').str.upper().str.contains(DRUG_PATTERN, regex=True)
    return set(med.loc[mask, 'SEQN'])


def load_and_merge_data(dataset_dir=None):
    dataset_dir = dataset_dir or find_dataset_dir()
    demo = read_table(dataset_dir, 'demographic',
                      ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'SDDSRVYR', 'RIDEXPRG', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA'])
    if not demo.SDDSRVYR.eq(8).all():
        raise ValueError('This pipeline is defined for NHANES 2013–2014 (cycle 8) only')
    adults = demo.loc[demo.RIDAGEYR.ge(20)].copy()
    audit = {'demographic_rows': len(demo), 'adults_before_exclusions': len(adults),
             'known_pregnant_excluded': int(adults.RIDEXPRG.eq(1).sum()), 'tables': {}}
    # Nonpregnant target thresholds; unknown pregnancy remains a documented limitation.
    merged = adults.loc[~adults.RIDEXPRG.eq(1)].copy()
    specs = {
        'examination': ['BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXSY1', 'BPXDI1',
                        'BPXSY2', 'BPXDI2', 'BPXSY3', 'BPXDI3'],
        'labs': ['LBXGH', 'LBXSGL'],
        'diet': ['DR1TKCAL', 'DR1TSUGR', 'DR1TFIBE', 'DR1DRSTZ'],
        'questionnaire': ['DIQ010', 'DIQ050', 'DIQ070', 'BPQ020', 'SMQ020', 'PAQ650', 'SLQ050'],
    }
    for name, cols in specs.items():
        table = read_table(dataset_dir, name, ['SEQN'] + cols)
        audit['tables'][name] = {'rows': len(table), 'matched_adults': int(merged.SEQN.isin(table.SEQN).sum())}
        merged = merged.merge(table, how='left', on='SEQN', validate='one_to_one')
    merged['on_diabetic_med'] = merged.SEQN.isin(get_diabetic_medication_seqns(dataset_dir)).astype(int)
    audit['merged_rows'] = len(merged)
    audit['unknown_pregnancy_status_retained'] = int(merged.RIDEXPRG.isna().sum() + merged.RIDEXPRG.eq(3).sum())
    audit['source_files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in sorted(Path(dataset_dir).glob('*.csv'))}
    merged.attrs['audit'] = audit
    return merged


def define_target(df):
    """Research proxy: HbA1c/diagnosis/therapy; never treat LBXSGL as fasting glucose.
    Controls require an observed HbA1c <6.5 and explicit DIQ010=2.
    HbA1c 5.7–6.4 remains negative for THIS binary diabetes proxy, not 'healthy'.
    Medication-only positives are a proxy and may include other indications.
    """
    df = df.copy()
    a1c = df.LBXGH.where(df.LBXGH.gt(0))
    evidence = pd.DataFrame({
        'a1c_positive': a1c.ge(6.5), 'diagnosis_positive': df.DIQ010.eq(1),
        'insulin_positive': df.DIQ050.eq(1), 'pills_positive': df.DIQ070.eq(1),
        'medication_positive': df.on_diabetic_med.eq(1),
    })
    positive = evidence.any(axis=1)
    negative = df.DIQ010.eq(2) & a1c.notna() & a1c.lt(6.5) & ~positive
    target = pd.Series(np.nan, index=df.index)
    target.loc[negative] = 0
    target.loc[positive] = 1
    df['is_diabetic'] = target
    audit = df.attrs.get('audit', {}).copy()
    old_positive = df.LBXGH.ge(6.5) | df.LBXSGL.ge(126) | df.DIQ010.eq(1) | df.on_diabetic_med.eq(1)
    audit.update({'label_evidence_counts_overlapping': evidence.sum().astype(int).to_dict(),
                  'lbxs_gl_126_without_corrected_positive': int((df.LBXSGL.ge(126) & ~positive).sum()),
                  'old_glucose_rule_positives_with_new_medication_parser': int(old_positive.sum()),
                  'indeterminate_excluded': int(target.isna().sum()),
                  'negative_with_a1c_5_7_to_6_4': int((negative & a1c.ge(5.7)).sum())})
    df = df.loc[target.notna()].copy()
    df['is_diabetic'] = df.is_diabetic.astype(int)
    audit.update({'labeled_rows': len(df), 'positive': int(df.is_diabetic.sum()),
                  'negative': int(df.is_diabetic.eq(0).sum())})
    df.attrs['audit'] = audit
    return df


def safe_sugar_fiber_ratio(sugar, fiber):
    """No ratio when fiber is zero/missing: do not generate values of 60 million."""
    return sugar / fiber.where(fiber.gt(0))


def hypertension_indicator(history, systolic, diastolic):
    """Keep unknown history unknown unless observed BP gives positive evidence."""
    flag = history.map({1: 1.0, 2: 0.0})
    return flag.mask(systolic.ge(140) | diastolic.ge(90), 1.0)


def engineer_features(df):
    df = df.copy()
    df['age'] = df.RIDAGEYR.clip(upper=80)  # 80 is top-coded, not exact age.
    df['gender_is_female'] = df.RIAGENDR.map({1: 0.0, 2: 1.0})
    for raw, feat in [('BMXWT', 'weight_kg'), ('BMXHT', 'height_cm'), ('BMXWAIST', 'waist_cm')]:
        df[feat] = df[raw].where(df[raw].gt(0))
    # Same exact formula for training and web input (not rounded BMI from file).
    df['bmi'] = df.weight_kg / (df.height_cm / 100).pow(2)
    df['waist_to_height_ratio'] = df.waist_cm / df.height_cm
    for prefix, feat in [('BPXSY', 'systolic_bp'), ('BPXDI', 'diastolic_bp')]:
        readings = df[[f'{prefix}{i}' for i in (1, 2, 3)]]
        # Use available positive readings. Zero diastolic is not imputed as normal BP.
        df[feat] = readings.where(readings.gt(0)).mean(axis=1)
    reliable = df.DR1DRSTZ.eq(1)
    for raw, feat in [('DR1TSUGR', 'daily_sugar_g'), ('DR1TFIBE', 'daily_fiber_g'), ('DR1TKCAL', 'daily_calories_kcal')]:
        df[feat] = df[raw].where(reliable & df[raw].ge(0))
    df['sugar_to_fiber_ratio'] = safe_sugar_fiber_ratio(df.daily_sugar_g, df.daily_fiber_g)
    df['hypertension_flag'] = hypertension_indicator(df.BPQ020, df.systolic_bp, df.diastolic_bp)
    df['no_vigorous_recreation_flag'] = df.PAQ650.map({1: 0.0, 2: 1.0})
    df['smoking_history_flag'] = df.SMQ020.map({1: 1.0, 2: 0.0})
    df['sleep_trouble_flag'] = df.SLQ050.map({1: 1.0, 2: 0.0})
    return df


def describe_features(X):
    return {c: {'median': float(X[c].median()), 'mean': float(X[c].mean()),
                'min': float(X[c].min()), 'max': float(X[c].max()),
                'p25': float(X[c].quantile(.25)), 'p75': float(X[c].quantile(.75)),
                'missing_n': int(X[c].isna().sum())} for c in FEATURE_COLUMNS}


def build_preprocessor():
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scaler', RobustScaler())]), NUMERICAL_FEATURES),
        ('bin', SimpleImputer(strategy='most_frequent', keep_empty_features=True), BINARY_FEATURES),
    ], remainder='drop')


def get_prepared_data(dataset_dir=None):
    df = engineer_features(define_target(load_and_merge_data(dataset_dir)))
    X = df.set_index('SEQN')[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df.set_index('SEQN').is_diabetic
    X.attrs['audit'] = df.attrs['audit']
    return X, y, describe_features(X)


if __name__ == '__main__':
    X, y, _ = get_prepared_data()
    print(X.attrs['audit'])
    print(X.isna().sum())
