"""Reproduce audit comparisons and API examples without fitting or selecting models."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from ml_pipeline.preprocess import ROOT, ARTIFACTS, load_and_merge_data, define_target, get_diabetic_medication_seqns
from ml_pipeline.evaluate import classification_metrics


def run():
    audit=ROOT/'audit'
    source=audit/'original/ml_pipeline/preprocess.py'
    spec=importlib.util.spec_from_file_location('historical_preprocess',source)
    original=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    old=original.define_target(original.load_and_merge_data(str(ROOT/'Dataset'))).set_index('SEQN').is_diabetic
    merged=load_and_merge_data()
    new=define_target(merged).set_index('SEQN').is_diabetic
    common=old.index.intersection(new.index)
    med=pd.read_csv(ROOT/'Dataset/medications.csv',encoding='latin1',low_memory=False)
    old_med=original.get_diabetic_medication_seqns(str(ROOT/'Dataset'))
    new_med=get_diabetic_medication_seqns(str(ROOT/'Dataset'))
    details={'original_labeled':len(old),'original_positive':int(old.sum()),'corrected_labeled':len(new),
        'corrected_positive':int(new.sum()),'common_subjects':len(common),
        'common_old_positive_new_negative':int(((old.loc[common]==1)&(new.loc[common]==0)).sum()),
        'common_old_negative_new_positive':int(((old.loc[common]==0)&(new.loc[common]==1)).sum()),
        'old_labeled_now_excluded':len(old.index.difference(new.index)),
        'newly_included':len(new.index.difference(old.index)),
        'medication_parser_old_subjects':len(old_med),'medication_parser_new_subjects':len(new_med),
        'old_only_medication_subjects':len(old_med-new_med),
        'drugs_for_old_only_medication_subjects':sorted(med.loc[med.SEQN.isin(old_med-new_med),'RXDDRUG'].dropna().unique().tolist())}
    (audit/'label_comparison.json').write_text(json.dumps(details,indent=2))
    from backend.app.main import app
    client=TestClient(app)
    baseline={'age':55,'gender':'male','weight_kg':90,'height_cm':172,'waist_cm':104,
              'systolic_bp':138,'diastolic_bp':86,'daily_sugar_g':95,'daily_fiber_g':12,
              'hypertension_history':True,'smoking_history':False,'vigorous_activity':False,'sleep_trouble':True}
    requests={'predict':('/api/predict',baseline),
        'simulate':('/api/simulate',{'baseline':baseline,'adjustments':{'delta_weight_kg':-5,'delta_waist_cm':-4,'delta_sugar_pct':-30,'delta_fiber_g':8,'active_exercise':True,'manage_hypertension':True}}),
        'identity_simulation':('/api/simulate',{'baseline':baseline,'adjustments':{}})}
    examples={}
    for name,(path,payload) in requests.items():
        response=client.post(path,json=payload)
        response.raise_for_status()
        examples[name]={'path':path,'request':payload,'response':response.json()}
    (audit/'api_examples.json').write_text(json.dumps(examples,indent=2))
    # Descriptive test subgroup checks, not model/threshold selection or fairness certification.
    import joblib
    split=joblib.load(ARTIFACTS/'train_test_splits.joblib')
    rows=pd.read_csv(ARTIFACTS/'test_predictions.csv')
    meta=json.loads((ARTIFACTS/'metadata.json').read_text())
    frame=rows.loc[rows.model.eq(meta['model_name'])].set_index('SEQN').join(split['X_test'][['age','gender_is_female']])
    groups={'female':frame.gender_is_female.eq(1),'male':frame.gender_is_female.eq(0),
        'age_20_39':frame.age.lt(40),'age_40_59':frame.age.ge(40)&frame.age.lt(60),'age_60_plus':frame.age.ge(60)}
    subgroup={}
    for name,mask in groups.items():
        f=frame.loc[mask]
        subgroup[name]={'n':len(f),'positive':int(f.y_true.sum()),
            'metrics':classification_metrics(f.y_true,f.probability,meta['decision_threshold']) if f.y_true.nunique()==2 else None}
    (audit/'subgroup_metrics.json').write_text(json.dumps(subgroup,indent=2))
    print(json.dumps(details,indent=2))


if __name__=='__main__':
    run()
