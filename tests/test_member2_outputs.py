"""Validate completed Member 2 runs and the public inference handoff."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

CARD=Path('models/model_card.json')
READY=CARD.exists() and json.loads(CARD.read_text()).get('status')=='final'
pytestmark=pytest.mark.skipif(not READY,reason='Complete python -m src.models.run_all first')


def test_five_families_and_complete_comparison():
    table=pd.read_csv('results/metrics/model_comparison.csv')
    assert {'logistic_regression','random_forest','hist_gradient_boosting','knn','mlp'}<=set(table.model_name)
    for column in ['cv_top1_accuracy','cv_macro_f1','student_top1_accuracy','student_macro_f1','student_log_loss','student_route_continuity']:
        assert np.isfinite(table[column]).all()
    experiments=pd.read_csv('results/metrics/experiments.csv')
    assert experiments.groupby('model_name').size().eq(6).all()
    assert Path('report/member2/models_and_tuning.md').exists()
    assert Path('results/plots/model_calibration.png').exists()


def test_real_tpe_trials_and_full_row_confirmation():
    summary=json.loads(Path('results/metrics/tuning_best.json').read_text())
    trials=pd.read_csv('results/metrics/tuning_trials.csv')
    assert len(trials)==summary['n_trials']>=4
    assert trials.state.eq('COMPLETE').all()
    assert trials.proposal.eq('TPE').sum()==summary['n_tpe_trials']>0
    train=pd.read_parquet('data/processed/train.parquet',columns=['trip_id','is_ref'])
    mapping=json.loads(Path('configs/folds.json').read_text())['trip_to_fold']
    assignments=train.trip_id.map(mapping)
    confirmation=json.loads(Path('results/metrics/tuning_confirmation.json').read_text())
    for row in confirmation['folds']:
        expected=int(((assignments!=row['fold']) & (train.is_ref>=0)).sum())
        assert row['n_fit_rows']==expected and row['n_fit_trips']==64
    for number in trials.trial:
        result=json.loads(Path(f'results/metrics/member2/trial_{number}.json').read_text())
        assert len(result['folds'])==5
        for row in result['folds']:
            assert row['n_fit_trips']==64
            # Proportional rounding/minimum-per-stratum can slightly exceed target.
            assert row['n_fit_rows']<=summary['search_max_train_rows']+128


def test_final_selection_calibration_and_export_limits():
    selection=json.loads(Path('results/metrics/model_selection.json').read_text())
    assert selection['student_test_used_for_selection'] is False
    table=pd.read_csv('results/metrics/model_comparison.csv')
    baseline=table[~table.model_name.str.endswith('_tuned')].sort_values(['cv_top1_accuracy','cv_macro_f1','model_name'],ascending=[False,False,True]).iloc[0]
    assert baseline.model_name==selection['baseline_winner']
    tuned=table[table.model_name==baseline.model_name+'_tuned'].iloc[0]
    assert selection['tuned_selected']==bool(tuned.cv_top1_accuracy>baseline.cv_top1_accuracy)
    card=json.loads(CARD.read_text())
    known_rows=int((pd.read_parquet('data/processed/train.parquet',columns=['is_ref']).is_ref>=0).sum())
    assert card['train_trips']==80 and card['train_rows']==known_rows
    audit=json.loads(Path('results/metrics/calibration_audit.json').read_text())
    populations=[set(audit[k]) for k in ['base_fit_trip_ids','calibration_trip_ids','audit_trip_ids']]
    assert list(map(len,populations))==[48,16,16]
    assert not populations[0]&populations[1] and not populations[0]&populations[2] and not populations[1]&populations[2]
    if selection['calibration_enabled']:
        assert card['calibration']['slope']>0
        assert audit['calibrated']['log_loss']<audit['raw']['log_loss']
        assert audit['calibrated']['brier_score']<=audit['raw']['brier_score']
    for path in Path('models').rglob('*.joblib'):
        assert path.stat().st_size<100_000_000,path


def test_final_scorer_fresh_process_has_no_reference_io():
    script='''import builtins,io,numpy as np,pandas as pd
from src.models.training import load_scorer
frame=pd.read_parquet('tests/fixtures/member1/candidates.parquet')
original=builtins.open
def guard(path,*args,**kwargs):
    name=str(path)
    assert not name.endswith('.route') and '/data/raw/' not in name and not name.startswith('data/raw/')
    return original(path,*args,**kwargs)
builtins.open=guard
io.open=guard
scorer=load_scorer('models')
score=scorer.predict_proba(frame)
assert score.shape==(len(frame),)
assert np.isfinite(np.log(score)).all() and np.isfinite(np.log1p(-score)).all()
without_labels=frame.drop(columns=['is_ref'])
assert np.array_equal(score,scorer.predict_proba(without_labels))
assert scorer.predict_proba(frame.iloc[:0]).shape==(0,)
'''
    subprocess.run([sys.executable,'-c',script],check=True)
