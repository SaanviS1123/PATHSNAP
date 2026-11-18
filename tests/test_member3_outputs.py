import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.models.training import file_hash,ROOT

READY=Path('results/metrics/member3_run.json').exists()
pytestmark=pytest.mark.skipif(not READY,reason='Run Member 3 evaluation first')


def test_frozen_decoder_and_disjoint_tuning_population():
    config=json.loads(Path('configs/hmm.json').read_text())
    model=json.loads(Path('results/metrics/hmm_tuning_model.json').read_text())
    populations=[set(model[k]) for k in ['base_fit_trip_ids','calibration_trip_ids','tuning_trip_ids']]
    assert list(map(len,populations))==[48,16,16]
    assert not populations[0]&populations[1] and not populations[0]&populations[2] and not populations[1]&populations[2]
    assert config['selection']['student_test_used'] is False
    assert config['selection']['n_configurations']==12
    for path,expected in config['inputs'].items():
        if Path(path).exists():
            assert file_hash(path)==expected,path
    grid=pd.read_csv('results/metrics/hmm_tuning.csv')
    best=grid.sort_values(['pointwise_accuracy','route_continuity','n_restarts','transition_beta_m','gap_restart_s'],ascending=[False,False,True,False,True]).iloc[0]
    assert config['parameters']['transition_beta_m']==best.transition_beta_m
    assert config['parameters']['gap_restart_s']==best.gap_restart_s


def test_full_comparison_and_latency_outputs():
    frame=pd.read_csv('results/metrics/hmm_comparison_by_trip.csv',dtype={'trip_id':str})
    student=frame[frame.partition=='student_test']
    assert student.groupby('variant').trip_id.nunique().eq(10).all()
    summary=pd.read_csv('results/metrics/hmm_comparison.csv')
    sample=summary[summary.partition=='student_test']
    assert sample.n_obs.eq(19160).all() and sample.n_known_obs.eq(19143).all()
    assert sample.n_unknown_obs.eq(17).all()
    for row in sample.itertuples():
        group=student[student.variant==row.variant]
        assert row.pointwise_accuracy==pytest.approx(group.n_correct.sum()/group.n_known_obs.sum())
        assert 0<=row.sequence_f1<=1 and 0<=row.route_continuity<=1
    pivot=student.pivot(index='trip_id',columns='variant',values='pointwise_accuracy')
    assert np.array_equal(pivot.classifier.to_numpy(),pivot.classifier_bridged.to_numpy())
    timing=pd.read_csv('results/metrics/decode_timing.csv')
    assert timing[timing.partition=='student_test'].n_obs.max()==3958
    assert timing[timing.partition=='train_speed_only'].n_obs.max()==19313
    assert (timing.total_ms>0).all() and (timing.warm_viterbi_ms>0).all()
    assert Path('report/member3/hmm_and_maps.md').exists()
    for trip in student.trip_id.unique():
        text=Path(f'maps/pred_{trip}.html').read_text()
        assert 'Reference route (offline report only)' in text
        assert 'Classifier-only arc choices' in text and 'HMM route with directed bridges' in text


@pytest.mark.skipif(not Path('data/processed/member3/prediction_00000056.parquet').exists(),reason='Private prediction cache not available')
def test_every_student_observation_and_road_bridge_is_valid():
    from src.hmm.network import RoadNetwork
    from src.hmm.predict import load_params
    from src.hmm.viterbi import route_segments
    params=load_params()
    for path in sorted(Path('data/processed/member3').glob('prediction_*.parquet')):
        trip_id=path.stem.split('_')[-1]
        network=RoadNetwork.from_trip_dir(ROOT/'data/raw/student_test'/trip_id)
        prediction=pd.read_parquet(path)
        assert np.array_equal(prediction.obs_idx,np.arange(len(network.xy)))
        assert prediction.hmm_arc.ge(0).all() and prediction.hmm_arc.lt(len(network.arcs)).all()
        segments=route_segments(prediction[['obs_idx','hmm_arc']].rename(columns={'hmm_arc':'arc_id'}),network,params)
        for segment in segments:
            assert np.array_equal(network.arcs[segment[:-1],1],network.arcs[segment[1:],0])
