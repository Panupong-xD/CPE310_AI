"""Serializable predictor for SHAP on the actual calibrated probability output."""
import numpy as np
import pandas as pd
from ml_pipeline.preprocess import FEATURE_COLUMNS


class ProbabilityPredictor:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        frame = pd.DataFrame(np.asarray(X), columns=FEATURE_COLUMNS)
        return self.model.predict_proba(frame)[:, 1]


def compute_sample_shap(explainer, feature_names, X):
    values = explainer(np.asarray(X), max_evals=2 * len(feature_names) + 1, silent=True)
    return {name: float(v) for name, v in zip(feature_names, values.values[0])}
