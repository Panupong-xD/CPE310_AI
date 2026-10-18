"""Scientific/data invariants and deployed-artifact regression checks (unittest)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fastapi.testclient import TestClient
from ml_pipeline.preprocess import (ARTIFACTS, FEATURE_COLUMNS, build_preprocessor, define_target,
    engineer_features, get_diabetic_medication_seqns, hypertension_indicator, get_prepared_data)
from ml_pipeline.evaluate import classification_metrics, select_threshold
from ml_pipeline.train import benchmark_models


class ScientificInvariants(unittest.TestCase):
    def test_target_does_not_use_nonfasting_glucose_or_missing_a1c_as_control(self):
        df=pd.DataFrame({'LBXGH':[5.5,np.nan,6.5,6.1,5.1,5.0],
            'LBXSGL':[190,90,90,90,90,90], 'DIQ010':[2,2,2,2,1,2],
            'DIQ050':[2,2,2,2,2,1], 'DIQ070':[2]*6, 'on_diabetic_med':[0]*6})
        labeled=define_target(df)
        self.assertNotIn(1,labeled.index)
        self.assertEqual(labeled.is_diabetic.tolist(),[0,1,0,1,1])

    def test_medication_match_is_not_an_indication_or_substring(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({'SEQN':[1,2,3,4,5], 'RXDUSE':[1,1,1,1,2],
                'RXDDRUG':['METFORMIN; SITAGLIPTIN','GLUCOSAMINE','OLMESARTAN','INSULIN GLARGINE','METFORMIN'],
                'RXDRSD1':['Other','Other','Type 2 diabetes mellitus','Other','Other']}).to_csv(Path(d)/'medications.csv',index=False)
            self.assertEqual(get_diabetic_medication_seqns(d),{1,4})

    def test_hypertension_unknown_not_treated_as_no(self):
        flag=hypertension_indicator(pd.Series([9.,7.,2.,1.]),pd.Series([120.,150.,120.,120.]),pd.Series([80.]*4))
        self.assertTrue(pd.isna(flag.iloc[0]))
        self.assertEqual(flag.iloc[1:].tolist(),[1.,0.,1.])

    def test_fold_preprocessing_fits_only_training_rows(self):
        rng=np.random.default_rng(10)
        X=pd.DataFrame(rng.normal(size=(50,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS)
        X.iloc[::4,1]=np.nan
        y=pd.Series([0,1]*25)
        fits=[]
        class RecordingPipeline(Pipeline):
            def fit(self,X,y,**kwargs):
                super().fit(X,y,**kwargs)
                actual=self.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer'].statistics_
                np.testing.assert_allclose(actual,np.nanmedian(X.iloc[:,:12],axis=0))
                fits.append(len(X))
                return self
        def model(clf):
            return RecordingPipeline([('preprocessor',build_preprocessor()),('classifier',clf)])
        with patch('ml_pipeline.train.get_candidate_models',return_value={
                'DummyPrior':DummyClassifier(strategy='prior'),'LogisticRegression':LogisticRegression()}), \
             patch('ml_pipeline.train.build_model',side_effect=model):
            benchmark_models(X,y)
        self.assertEqual(fits.count(40),10)
        self.assertEqual(fits.count(50),2)

    def test_threshold_and_confusion_matrix(self):
        y=[0,0,1,1]
        p=[.1,.2,.3,.9]
        t=select_threshold(y,p)
        self.assertEqual(t,.3)
        self.assertEqual(classification_metrics(y,p,t)['confusion_matrix'],{'tn':2,'fp':0,'fn':0,'tp':2})


class ArtifactIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.app.main import app
        from backend.app.model_service import model_service
        cls.client=TestClient(app)
        cls.service=model_service
        cls.splits=joblib.load(ARTIFACTS/'train_test_splits.joblib')
        cls.input={'age':55,'gender':'male','weight_kg':90,'height_cm':172,'waist_cm':104,
                   'systolic_bp':138,'diastolic_bp':86,'daily_sugar_g':95,'daily_fiber_g':12,
                   'hypertension_history':True,'smoking_history':False,'vigorous_activity':True,'sleep_trouble':True}

    def test_disjoint_subjects_and_fit_only_statistics(self):
        s=self.splits
        groups=[set(s[k].index) for k in ['X_train','X_cal','X_test']]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        for col,stats in s['feature_stats'].items():
            self.assertEqual(stats['median'],s['X_train'][col].median())
        self.assertFalse(set(FEATURE_COLUMNS)&{'SEQN','LBXGH','LBXSGL','DIQ010','DIQ050','DIQ070','on_diabetic_med'})

    def test_training_and_api_feature_engineering_are_identical(self):
        from backend.app.schemas import AssessmentInput
        X,y,_=get_prepared_data()
        complete=X.dropna()
        row=complete.iloc[0]
        inp=AssessmentInput(age=row.age,gender='female' if row.gender_is_female else 'male',
            weight_kg=row.weight_kg,height_cm=row.height_cm,waist_cm=row.waist_cm,
            systolic_bp=row.systolic_bp,diastolic_bp=row.diastolic_bp,
            daily_sugar_g=row.daily_sugar_g,daily_fiber_g=row.daily_fiber_g,daily_calories_kcal=row.daily_calories_kcal,
            hypertension_history=bool(row.hypertension_flag),vigorous_activity=not bool(row.no_vigorous_recreation_flag),
            smoking_history=bool(row.smoking_history_flag),sleep_trouble=bool(row.sleep_trouble_flag))
        actual=self.service._input_to_dataframe(inp)
        np.testing.assert_allclose(actual.iloc[0],row,atol=1e-12)

    def test_predict_shap_reconstructs_and_outputs_real_signs(self):
        res=self.client.post('/api/predict',json=self.input)
        self.assertEqual(res.status_code,200,res.text)
        self.assertTrue(self.client.get('/health').json()['model_loaded'])
        self.assertEqual(self.client.get('/api/metadata').json()['schema_version'],2)
        r=res.json()
        self.assertAlmostEqual(r['shap_base_score']+r['shap_total_contribution_points'],r['risk_score'],delta=.051)
        for factor in r['key_drivers']:
            self.assertGreater(factor['impact_value'],0)
            self.assertTrue(factor['impact'].endswith(' pp'))
        for factor in r['protective_factors']:
            self.assertLess(factor['impact_value'],0)
        self.assertEqual(r['imputed_fields'],['daily_calories_kcal'])

    def test_missing_inputs_are_missing_not_invented_measurements(self):
        res=self.client.post('/api/predict',json={'age':90,'gender':'male','height_cm':170,'weight_kg':70})
        self.assertEqual(res.status_code,200,res.text)
        r=res.json()
        self.assertIsNone(r['waist_to_height_ratio'])
        self.assertIn('waist_cm',r['imputed_fields'])
        self.assertIn('systolic_bp',r['imputed_fields'])

    def test_zero_fiber_and_invalid_age(self):
        res=self.client.post('/api/predict',json={**self.input,'daily_fiber_g':0})
        self.assertEqual(res.status_code,200,res.text)
        self.assertIn('sugar_to_fiber_ratio',res.json()['imputed_fields'])
        self.assertEqual(self.client.post('/api/predict',json={**self.input,'age':19}).status_code,422)
        without_gender={k:v for k,v in self.input.items() if k!='gender'}
        self.assertEqual(self.client.post('/api/predict',json=without_gender).status_code,422)

    def test_identity_simulation_and_bp_history(self):
        r=self.client.post('/api/simulate',json={'baseline':self.input,'adjustments':{}})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['delta_risk_score'],0)
        from backend.app.schemas import SimulationRequest
        captured=[]
        original=self.service.predict
        def record(inp):
            captured.append(inp)
            return original(inp)
        with patch.object(self.service,'predict',side_effect=record):
            self.service.simulate(SimulationRequest(baseline=self.input,adjustments={'manage_hypertension':True}))
        self.assertTrue(captured[-1].hypertension_history)
        self.assertEqual(captured[-1].vigorous_activity,True)

    def test_saved_test_predictions_reproduce_metrics(self):
        rows=pd.read_csv(ARTIFACTS/'test_predictions.csv')
        meta=self.service.metadata
        for name,frame in rows.groupby('model'):
            recorded=meta['test_benchmarks'][name]
            actual=classification_metrics(frame.y_true,frame.probability,recorded['threshold'])
            self.assertEqual(actual['confusion_matrix'],recorded['confusion_matrix'])
            self.assertAlmostEqual(actual['roc_auc'],recorded['roc_auc'],places=12)
        self.assertEqual(meta['model_name'],max((n for n in meta['cv_benchmarks'] if n!='DummyPrior'),
            key=lambda n:meta['cv_benchmarks'][n]['roc_auc_mean']))


if __name__=='__main__':
    unittest.main(verbosity=2)
