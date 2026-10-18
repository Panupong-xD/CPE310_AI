"""Independent sigmoid calibration, locked test comparison and probability SHAP."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    recall_score, precision_score, f1_score, confusion_matrix, brier_score_loss,
    log_loss, roc_curve, precision_recall_curve)
from ml_pipeline.preprocess import ARTIFACTS, FEATURE_COLUMNS
from ml_pipeline.explain import ProbabilityPredictor, compute_sample_shap
from ml_pipeline.train import SEED


def classification_metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'threshold': float(threshold), 'roc_auc': float(roc_auc_score(y,p)),
        'average_precision': float(average_precision_score(y,p)),
        'sensitivity': float(recall_score(y,pred,zero_division=0)),
        'specificity': float(tn/(tn+fp)), 'precision': float(precision_score(y,pred,zero_division=0)),
        'f1': float(f1_score(y,pred,zero_division=0)), 'accuracy': float(accuracy_score(y,pred)),
        'brier': float(brier_score_loss(y,p)), 'log_loss': float(log_loss(y,p,labels=[0,1])),
        'confusion_matrix': {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}}


def select_threshold(y, probabilities, sensitivity_target=.8):
    # Highest specificity subject to >= target recall on fit-only OOF predictions.
    fpr, tpr, thresholds = roc_curve(y, probabilities, drop_intermediate=False)
    valid = np.flatnonzero((tpr >= sensitivity_target) & np.isfinite(thresholds))
    i = min(valid, key=lambda k: (fpr[k], -thresholds[k]))
    return float(thresholds[i])


def bootstrap_intervals(y, p, threshold, repeats=1000):
    # Stratified IID bootstrap: conditional on this fitted model and observed sample.
    # Not survey/design-aware and excludes training/selection uncertainty.
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == c) for c in (0,1)]
    draws = {k: [] for k in ('roc_auc','average_precision','sensitivity','specificity')}
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        yy, pp = y[ix], p[ix]
        draws['roc_auc'].append(roc_auc_score(yy,pp))
        draws['average_precision'].append(average_precision_score(yy,pp))
        draws['sensitivity'].append(float((pp[yy==1] >= threshold).mean()))
        draws['specificity'].append(float((pp[yy==0] < threshold).mean()))
    return {k: [float(v) for v in np.quantile(a,[.025,.975])] for k,a in draws.items()}


def run_evaluation(artifacts_dir=ARTIFACTS):
    out = Path(artifacts_dir)
    splits = joblib.load(out / 'train_test_splits.joblib')
    X_fit,y_fit = splits['X_train'],splits['y_train']
    X_cal,y_cal = splits['X_cal'],splits['y_cal']
    X_test,y_test = splits['X_test'],splits['y_test']
    best = splits['best_model_name']  # Already selected; never rerank using test results.
    benchmarks, calibrated_models, oof_probs, prediction_rows = {}, {}, {}, []
    curves = {}
    for name, raw in splits['fitted_models'].items():
        calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method='sigmoid', ensemble=False)
        calibrated.fit(X_cal,y_cal)
        # sklearn uses decision_function for LR and predict_proba for trees.
        sigmoid = calibrated.calibrated_classifiers_[0].calibrators[0]
        oof = sigmoid.predict(splits['oof'][name]['responses'])
        threshold = select_threshold(y_fit,oof)
        p = calibrated.predict_proba(X_test)[:,1]
        raw_p = raw.predict_proba(X_test)[:,1]
        metrics = classification_metrics(y_test,p,threshold)
        metrics['at_0_5'] = classification_metrics(y_test,p,.5)
        metrics['raw_uncalibrated'] = classification_metrics(y_test,raw_p,.5)
        metrics['ci95_stratified_iid_bootstrap'] = bootstrap_intervals(y_test,p,threshold)
        metrics['threshold_fit_oof'] = classification_metrics(y_fit,oof,threshold)
        benchmarks[name] = metrics
        calibrated_models[name],oof_probs[name] = calibrated,oof
        fpr,tpr,rt=roc_curve(y_test,p)
        precision,recall,pt=precision_recall_curve(y_test,p)
        obs,pred=calibration_curve(y_test,p,n_bins=8,strategy='quantile')
        curves[name] = {'roc': {'fpr': fpr.tolist(),'tpr':tpr.tolist()},
                        'precision_recall': {'precision':precision.tolist(),'recall':recall.tolist()},
                        'calibration': {'mean_predicted':pred.tolist(),'fraction_positive':obs.tolist()}}
        prediction_rows.extend({'SEQN': int(seqn), 'model':name, 'y_true':int(yy),
            'probability':float(pp),'raw_probability':float(rp),'threshold':threshold,'prediction':int(pp>=threshold)}
            for seqn, yy, pp, rp in zip(X_test.index,y_test,p,raw_p))
        print(f'{name}: test AUC={metrics["roc_auc"]:.4f}, AP={metrics["average_precision"]:.4f}, recall={metrics["sensitivity"]:.4f}, threshold={threshold:.4f}',flush=True)
    pipeline = calibrated_models[best]
    background = X_fit.sample(n=min(32,len(X_fit)),random_state=SEED)
    explainer = shap.PermutationExplainer(ProbabilityPredictor(pipeline), background.to_numpy(),
        feature_names=FEATURE_COLUMNS, seed=SEED)
    sample = X_fit.sample(n=min(100,len(X_fit)),random_state=SEED+1)
    explanation = explainer(sample.to_numpy(),max_evals=2*len(FEATURE_COLUMNS)+1,silent=True)
    explained_probability = explanation.base_values + explanation.values.sum(axis=1)
    np.testing.assert_allclose(explained_probability,pipeline.predict_proba(sample)[:,1],atol=1e-7)
    importance = sorted([{'feature':c,'importance':float(v)} for c,v in
        zip(FEATURE_COLUMNS,np.abs(explanation.values).mean(axis=0))],key=lambda r:r['importance'],reverse=True)
    ref = oof_probs[best]
    m = benchmarks[best]
    metadata = {'schema_version':2,'model_name':best,
        'task':'Cross-sectional diabetes proxy classification; not incident diabetes or prediabetes diagnosis',
        'metrics':{'test_roc_auc':m['roc_auc'],'test_pr_auc':m['average_precision'],
            'test_recall_sensitivity':m['sensitivity'],'test_specificity':m['specificity'],
            'test_precision':m['precision'],'test_f1':m['f1'],'test_accuracy':m['accuracy'],
            'test_brier':m['brier'],'confusion_matrix':m['confusion_matrix']},
        'cv_benchmarks':splits['cv_results'],'test_benchmarks':benchmarks,
        'feature_names':FEATURE_COLUMNS,'raw_feature_columns':FEATURE_COLUMNS,
        'feature_stats':splits['feature_stats'],'feature_stats_source':'fit partition only',
        'feature_importance':importance,'importance_method':'Permutation SHAP on calibrated probability; 32 fit background rows, 100 fit explanation rows, one forward/reverse permutation',
        'explanation_units':'probability; multiply by 100 for percentage points',
        'percentile_bins':np.quantile(ref,np.linspace(0,1,101)).tolist(),
        'percentile_source':'fit-only out-of-fold scores mapped through independent sigmoid calibration; unweighted sample reference',
        'decision_threshold':m['threshold'],'threshold_rule':'fit OOF sensitivity >=0.80, maximize specificity; no test tuning',
        'risk_thresholds':{'low_max':20.,'moderate_max':50.,'high_min':50.},
        'risk_categories':{'low':'Lower model score','moderate':'Intermediate model score','high':'Higher model score'},
        'risk_tiers_note':'20/50 percent are display bands, not validated diagnostic thresholds',
        'calibration':'Sigmoid fitted on independent 20% calibration partition; base models fitted on 60%; final 20% test',
        'test_prevalence':float(y_test.mean()),'test_n':len(y_test),
        'limitations':['Unweighted 2013–2014 US sample; not population prevalence or validated Thai risk',
            'Single-visit HbA1c/diagnosis/therapy proxy; not clinical confirmation',
            'Known diabetes included; dietary reverse causation and treatment effects possible',
            'Survey clusters and weights not modeled; IID CIs are approximate',
            'SHAP and what-if scores are associations, not intervention effects',
            'One split and preexisting historical test exposure; independent external validation still required']}
    joblib.dump(pipeline,out/'model_pipeline.joblib')
    joblib.dump({'explainer':explainer,'feature_names':FEATURE_COLUMNS,
                'base_value':float(explanation.base_values[0]),'output_space':'calibrated_probability'},out/'shap_explainer.joblib')
    joblib.dump(calibrated_models,out/'benchmark_models.joblib')
    pd.DataFrame(prediction_rows).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame({'SEQN':X_fit.index,'y_true':y_fit,'calibrated_oof_probability':ref}).to_csv(out/'champion_oof_predictions.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    (out/'evaluation_curves.json').write_text(json.dumps(curves,indent=2,allow_nan=False))
    return metadata


if __name__ == '__main__':
    run_evaluation()
