import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,ClassifierMixin
from src.evaluation.harness import make_group_splits,score_predictions,evaluate_model,evaluate_holdout,winners
from src.evaluation.continuity import route_continuity
from src.models.specs import make_estimator
from src.models.training import CandidateScorer,export_scorer,load_scorer,scorer_from_fitted
from src.models.run_all import select_winner
from src.models.calibration import SigmoidCalibrator


def test_ties_missing_candidates_and_unknown_labels():
    frame=pd.DataFrame(dict(trip_id=['x']*5,obs_idx=[0,0,1,2,2],arc_id=[8,2,3,5,6],is_ref=[0,1,0,-1,-1]))
    labels=pd.DataFrame(dict(trip_id=['x']*4,obs_idx=[0,1,2,3],ref_arc=[2,9,-1,10]))
    scores=np.array([.8,.8,.1,.3,.9])
    result=score_predictions(frame,scores,labels)
    assert result['top1_accuracy']==pytest.approx(1/3)
    assert result['candidate_miss_rate']==pytest.approx(2/3)
    assert result['n_known_candidate_rows']==3
    assert result['n_unknown_obs']==1
    assert winners(frame,scores).iloc[0].arc_id==2
    # Ties remain independent of input row order or repeated pandas indexes.
    reordered=frame.iloc[[1,0,2,3,4]].copy()
    reordered.index=[0]*5
    assert winners(reordered,scores[[1,0,2,3,4]]).iloc[0].arc_id==2


def test_continuity_directed_adjacency_and_missing_observations():
    arcs=np.array([[0,1],[1,2],[3,4],[2,1]])
    prediction=pd.DataFrame(dict(trip_id=['x']*5,obs_idx=[0,1,2,3,5],arc_id=[0,0,1,2,2]))
    assert route_continuity(prediction,{'x':arcs})==.5
    assert route_continuity(prediction.iloc[:1],{'x':arcs}) is None


def test_grouped_folds_are_frozen_and_complete(tmp_path):
    path=tmp_path/'folds.json'
    path.write_text(json.dumps({'trip_to_fold':{'a':0,'b':1,'c':2}}))
    groups=np.array(['a','a','b','b','c','c'])
    seen=[]
    for train,val in make_group_splits(groups,folds_path=path):
        assert not set(groups[train])&set(groups[val])
        seen.extend(val)
    assert sorted(seen)==list(range(6))
    with pytest.raises(ValueError,match='every trip'):
        list(make_group_splits(['a','d'],folds_path=path))


class AuditedClassifier(ClassifierMixin,BaseEstimator):
    fit_groups=[]
    def fit(self,X,y):
        assert set(y)=={0,1}
        assert (X['value']<100).all()  # Unknown-label rows must never enter fitting.
        self.__class__.fit_groups.append(sorted(set(X['group_marker'])))
        self.classes_=np.array([0,1])
        return self
    def predict_proba(self,X):
        p=np.where(X['value']>0,.8,.2)
        return np.column_stack([1-p,p])


def test_shared_cv_filters_unknowns_and_scores_all_observations(tmp_path):
    rows=[];labels=[]
    for group in range(3):
        trip=f'{group:08d}'
        rows.extend([dict(trip_id=trip,obs_idx=0,arc_id=0,is_ref=1,value=1.,group_marker=group),
                     dict(trip_id=trip,obs_idx=0,arc_id=1,is_ref=0,value=-1.,group_marker=group),
                     dict(trip_id=trip,obs_idx=1,arc_id=2,is_ref=-1,value=999.,group_marker=group)])
        labels.extend([dict(trip_id=trip,obs_idx=0,ref_arc=0),dict(trip_id=trip,obs_idx=1,ref_arc=-1)])
    data=pd.DataFrame(rows)
    path=tmp_path/'folds.json'
    path.write_text(json.dumps({'trip_to_fold':{f'{i:08d}':i for i in range(3)}}))
    oof=tmp_path/'oof.npy'
    AuditedClassifier.fit_groups=[]
    result=evaluate_model(AuditedClassifier(),'audit',data,['value','group_marker'],obs_labels=pd.DataFrame(labels),folds_path=path,oof_path=oof,log=False)
    assert result['top1_accuracy']==1
    assert np.isfinite(np.load(oof)).all()
    assert len(AuditedClassifier.fit_groups)==3
    assert all(len(groups)==2 for groups in AuditedClassifier.fit_groups)
    assert all(f['n_unknown_obs']==1 and f['n_candidate_rows']==3 for f in result['folds'])
    with pytest.raises(ValueError,match='overlap'):
        evaluate_holdout(AuditedClassifier(),data,data,pd.DataFrame(labels),['value','group_marker'])


def test_boosting_no_row_validation_and_bounded_forest():
    assert make_estimator('hist_gradient_boosting').early_stopping is False
    rf=make_estimator('random_forest')
    assert rf.max_depth==14 and rf.n_jobs==2 and rf.max_samples==.5


def test_selection_does_not_consult_student_test():
    table=pd.DataFrame(dict(model_name=['a','b'],cv_top1_accuracy=[.8,.7],cv_macro_f1=[.7,.9],student_top1_accuracy=[.1,1.]))
    assert select_winner(table)=='a'
    table.student_top1_accuracy=[1.,.1]
    assert select_winner(table)=='a'


def test_scorer_exports_preprocessing_once_and_checks_tampering(tmp_path):
    from src.features.build import FEATURES
    fixture=pd.read_parquet('tests/fixtures/member1/candidates.parquet')
    estimator=make_pipeline(StandardScaler(),LogisticRegression()).fit(fixture[FEATURES],fixture.is_ref)
    expected=estimator.predict_proba(fixture[FEATURES])[:,1]
    scorer=scorer_from_fitted(estimator,FEATURES)
    folder=tmp_path/'model'
    export_scorer(scorer,'test',FEATURES,folder,dict(status='test'))
    loaded=load_scorer(folder)
    assert np.allclose(loaded.predict_proba(fixture),expected)
    assert loaded.predict_proba(fixture.iloc[:0]).shape==(0,)
    reordered=fixture[list(reversed(fixture.columns))]
    assert np.allclose(loaded.predict_proba(reordered),expected)
    (folder/'preprocessor.joblib').write_bytes(b'broken')
    with pytest.raises(ValueError,match='checksum'):
        load_scorer(folder)


def test_sigmoid_calibrator_is_monotonic():
    scores=np.linspace(.01,.99,100)
    y=(scores>.6).astype(int)
    calibrator=SigmoidCalibrator().fit(scores,y)
    assert calibrator.increasing
    predicted=calibrator.predict(scores)
    assert (np.diff(predicted)>=0).all()
    assert np.isfinite(calibrator.predict([0.,1.])).all()


def _tiny_trips(n_trips):
    rows=[];labels=[]
    for trip in range(n_trips):
        for obs in range(10):
            for arc,y in [(0,1),(1,0)]:
                rows.append(dict(trip_id=str(trip),obs_idx=obs,arc_id=arc,is_ref=y,value=(1 if y else -1)*(1+obs/100)))
            labels.append(dict(trip_id=str(trip),obs_idx=obs,ref_arc=0))
    return pd.DataFrame(rows),pd.DataFrame(labels)


def test_tpe_budget_full_confirmation_and_resume(tmp_path,monkeypatch):
    pytest.importorskip('optuna')
    import src.models.tuning.bayes_opt as search
    import src.evaluation.harness as harness
    config=tmp_path/'configs/folds.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'trip_to_fold':{str(i):i for i in range(4)}}))
    monkeypatch.setattr(search,'ROOT',tmp_path)
    monkeypatch.setattr(harness,'FOLDS_PATH',config)
    monkeypatch.setattr(search,'route_continuity',lambda predictions:1.)
    monkeypatch.setattr(search,'input_signature',lambda features,extra=None:('unit-study',{'unit_test':True}))
    data,labels=_tiny_trips(4)
    summary,confirmation=search.tune(trials=4,model_name='logistic_regression',frame=data,features=['value'],obs_labels=labels,search_max_train_rows=20)
    assert summary['n_trials']==4 and summary['n_tpe_trials']==1
    trial=json.loads((tmp_path/'results/metrics/member2/trial_0.json').read_text())
    assert all(f['n_fit_trips']==3 and f['n_fit_rows']<=26 and f['n_candidate_rows']==20 for f in trial['folds'])
    assert all(f['n_fit_rows']==60 for f in confirmation['folds'])
    assert np.isfinite(np.load(tmp_path/summary['oof_path'])).all()
    resumed,_=search.tune(trials=4,model_name='logistic_regression',frame=data,features=['value'],obs_labels=labels,search_max_train_rows=20)
    assert resumed['n_trials']==4
    assert len(pd.read_csv(tmp_path/'results/metrics/tuning_trials.csv'))==4


def test_calibration_audit_uses_disjoint_training_trip_sets(tmp_path,monkeypatch):
    import src.models.calibration as calibration
    config=tmp_path/'configs/folds.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'trip_to_fold':{str(i):i for i in range(5)}}))
    monkeypatch.setattr(calibration,'ROOT',tmp_path)
    data,labels=_tiny_trips(5)
    accepted,audit=calibration.audit_calibration(make_pipeline(StandardScaler(),LogisticRegression()),data,['value'],labels)
    fitting=set(audit['base_fit_trip_ids'])
    calibration_trips=set(audit['calibration_trip_ids'])
    audit_trips=set(audit['audit_trip_ids'])
    assert not fitting&calibration_trips and not fitting&audit_trips and not calibration_trips&audit_trips
    assert len(fitting)==3 and len(calibration_trips)==len(audit_trips)==1
    assert isinstance(accepted,bool)


def test_legacy_csv_fields_are_normalized_without_duplicate_keywords():
    from src.models.run_all import flatten_experiments
    records=[dict(name='actual',cv={'folds':[dict(model_name='old',partition='old',fold=0,macro_f1=.8)]},student=dict(model_name='old',partition='old',macro_f1=.9))]
    rows=flatten_experiments(records)
    assert [row['model_name'] for row in rows]==['actual','actual']
    assert [row['partition'] for row in rows]==['cv','student_test']
