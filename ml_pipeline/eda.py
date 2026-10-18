"""Fit-partition exploratory summaries; never inspect test labels to select features."""
import json
from pathlib import Path
import joblib
from ml_pipeline.preprocess import ARTIFACTS, describe_features


def run_eda(output_dir=ARTIFACTS):
    out=Path(output_dir)
    split=joblib.load(out/'train_test_splits.joblib')
    X,y=split['X_train'],split['y_train']
    df=X.assign(is_diabetic=y)
    summary={'scope':'fit partition only, unweighted', 'dataset_size':len(X),
        'prevalence':float(y.mean()),'class_distribution':{'diabetic':int(y.sum()),'non_diabetic':int((1-y).sum())},
        'feature_correlations':df.corr().is_diabetic.drop('is_diabetic').sort_values(ascending=False).to_dict(),
        'group_means':df.groupby('is_diabetic').mean().T.rename(columns={0:'proxy_negative',1:'proxy_positive'}).to_dict(),
        'missing_percentages':(X.isna().mean()*100).to_dict(),'feature_stats':describe_features(X)}
    (out/'eda_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    run_eda()
