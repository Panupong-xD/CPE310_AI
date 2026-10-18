"""Fit-only CV selection; independent calibration and final test partitions."""
import json
import platform
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
from ml_pipeline.preprocess import ARTIFACTS, build_preprocessor, get_prepared_data, describe_features

SEED = 42


def get_candidate_models():
    # Fixed, declared candidates. This is benchmarking, NOT hyperparameter tuning.
    return {
        'DummyPrior': DummyClassifier(strategy='prior', random_state=SEED),
        'LogisticRegression': LogisticRegression(class_weight='balanced', C=.5, max_iter=3000, random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_split=10,
                          min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=SEED),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=250, max_depth=8, min_samples_split=10,
                          min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=SEED),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=180, learning_rate=.04,
                          max_depth=4, subsample=.85, random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=180, learning_rate=.04,
                          max_depth=5, class_weight='balanced', random_state=SEED),
    }


def build_model(classifier):
    return Pipeline([('preprocessor', build_preprocessor()), ('classifier', clone(classifier))])


def response_scores(model, X):
    # Match sklearn's calibration input: decision_function when available, else probability.
    return model.decision_function(X) if hasattr(model, 'decision_function') else model.predict_proba(X)[:, 1]


def benchmark_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, oof, fitted = {}, {}, {}
    for name, classifier in get_candidate_models().items():
        fold_results = []
        probabilities = np.full(len(y_train), np.nan)
        responses = np.full(len(y_train), np.nan)
        for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
            model = build_model(classifier)
            model.fit(X_train.iloc[tr], y_train.iloc[tr])
            p = model.predict_proba(X_train.iloc[va])[:, 1]
            probabilities[va] = p
            responses[va] = response_scores(model, X_train.iloc[va])
            yy = y_train.iloc[va]
            fold_results.append({'fold': fold, 'n_fit': len(tr), 'n_validation': len(va),
                'roc_auc': float(roc_auc_score(yy, p)),
                'average_precision': float(average_precision_score(yy, p)),
                'recall_at_0_5': float(recall_score(yy, p >= .5, zero_division=0)),
                'f1_at_0_5': float(f1_score(yy, p >= .5, zero_division=0))})
        results[name] = {'folds': fold_results}
        for metric in ('roc_auc', 'average_precision', 'recall_at_0_5', 'f1_at_0_5'):
            values = [f[metric] for f in fold_results]
            results[name][metric + '_mean'] = float(np.mean(values))
            results[name][metric + '_std'] = float(np.std(values, ddof=1))
        oof[name] = {'probabilities': probabilities, 'responses': responses}
        final = build_model(classifier).fit(X_train, y_train)
        fitted[name] = final
        print(f"{name}: CV AUC={results[name]['roc_auc_mean']:.4f}, AP={results[name]['average_precision_mean']:.4f}", flush=True)
    eligible = [n for n in results if n != 'DummyPrior']
    best = max(eligible, key=lambda n: results[n]['roc_auc_mean'])
    return best, results, fitted, oof


def train_and_export(output_dir=ARTIFACTS):
    output_dir = __import__('pathlib').Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, _ = get_prepared_data()
    # Reserve test before fitting or exploring training distributions.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=SEED)
    ids = {'fit': X_fit.index.tolist(), 'calibration': X_cal.index.tolist(), 'test': X_test.index.tolist()}
    assert not (set(ids['fit']) & set(ids['test']) or set(ids['fit']) & set(ids['calibration']) or set(ids['test']) & set(ids['calibration']))
    best, cv, fitted, oof = benchmark_models(X_fit, y_fit)
    splits = {'X_train': X_fit, 'y_train': y_fit, 'X_cal': X_cal, 'y_cal': y_cal,
              'X_test': X_test, 'y_test': y_test, 'feature_stats': describe_features(X_fit),
              'best_model_name': best, 'cv_results': cv, 'fitted_models': fitted, 'oof': oof}
    joblib.dump(splits, output_dir / 'train_test_splits.joblib')
    manifest = {'seed': SEED, 'selection_rule': 'Highest mean fit-only 5-fold ROC-AUC; test never selects model',
                'partitions': {k: {'n': len(v), 'positive': int(y.loc[v].sum()), 'seqn': v} for k,v in ids.items()},
                'data_audit': X.attrs['audit'],
                'versions': {'python': platform.python_version(), **{p: version(p) for p in ['numpy','pandas','scikit-learn','shap','joblib','fastapi']}}}
    (output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Selected {best}; fit={len(X_fit)}, calibration={len(X_cal)}, test={len(X_test)}', flush=True)
    return splits


if __name__ == '__main__':
    train_and_export()
