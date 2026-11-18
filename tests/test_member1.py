from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from src.data_processing.synthetic import write_fixture
from src.data_processing.loaders import Trip, load_trip, trip_directories
from src.data_processing.geo import project_trip
from src.features.candidates import CandidateIndex
from src.features.labels import align_labels
from src.features.build import FEATURES, build_features
from src.models.member1_harness import evaluate, fit_model


@pytest.fixture
def fixture(tmp_path):
    write_fixture(tmp_path)
    return tmp_path


def test_coordinate_order_and_metric_crs_across_regions(fixture):
    for folder in sorted(fixture.glob('0*')):
        trip = load_trip(folder)
        metric = project_trip(trip)
        _, _, distance = Geod(ellps='WGS84').inv(*trip.nodes[0], *trip.nodes[1])
        assert metric.length[0] == pytest.approx(distance, rel=1e-4)
        assert metric.xy.shape == (12, 2)


def test_inference_never_opens_route_and_is_label_independent(fixture, monkeypatch):
    original = np.loadtxt
    def guarded(path, *args, **kwargs):
        assert not str(path).endswith('.route')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'loadtxt', guarded)
    folder = fixture / '00000001'
    before = build_features(project_trip(load_trip(folder)), 50, 4)
    (folder / '00000001.route').unlink()
    after = build_features(project_trip(load_trip(folder)), 50, 4)
    pd.testing.assert_frame_equal(before, after)
    assert np.isfinite(before[FEATURES]).all().all()


def test_candidate_distance_is_to_segment_and_empty_radius_falls_back():
    nodes = np.array([[0., 0.], [.02, 0.], [.01, .001], [.01, .002]])
    trip = Trip('00000001', np.array([[.01, .00001, 0.], [.01, .01, 1.]]), nodes, np.array([[0, 1], [2, 3]]))
    metric = project_trip(trip)
    candidates = CandidateIndex(metric).query(10, 2)
    assert candidates[candidates.obs_idx == 0].iloc[0].arc_id == 0
    assert candidates[candidates.obs_idx == 0].iloc[0].distance_m < 2
    assert len(candidates[candidates.obs_idx == 1]) == 1
    assert candidates[candidates.obs_idx == 1].outside_radius.eq(1).all()
    assert candidates.groupby('obs_idx').size().max() <= 2


def test_label_alignment_preserves_repeated_route_occurrences():
    nodes = np.array([[0., 0.], [.001, 0.]])
    track = np.array([[.0002, 0, 0], [.0008, 0, 1], [.0008, 0, 2], [.0002, 0, 3], [.0002, 0, 4], [.0008, 0, 5]])
    trip = Trip('00000001', track, nodes, np.array([[0, 1], [1, 0]]), np.array([0, 1, 0]))
    labels = align_labels(project_trip(trip))
    assert (np.diff(labels.route_position) >= 0).all()
    assert labels.ref_arc.iloc[-1] == 0
    assert labels.route_position.iloc[-1] == 2
    assert labels.ref_arc.iloc[3] == 1


def test_unknown_label_does_not_become_negative():
    nodes = np.array([[0., 0.], [.001, 0.]])
    trip = Trip('00000001', np.array([[.0005, .01, 0.]]), nodes, np.array([[0, 1]]), np.array([0]))
    labels = align_labels(project_trip(trip))
    assert labels.ref_arc.tolist() == [-1]
    assert labels.label_known.tolist() == [False]


def test_metrics_count_misses_but_exclude_unknown_labels():
    table = pd.DataFrame(dict(trip_id=['x']*4, obs_idx=[0, 0, 1, 2], arc_id=[1, 2, 2, 3], is_ref=[1, 0, 0, -1]))
    labels = pd.DataFrame(dict(trip_id=['x']*3, obs_idx=[0, 1, 2], ref_arc=[1, 9, -1]))
    result = evaluate(table, labels, [.9, .1, .2, .9])
    assert result['top1_acc'] == .5
    assert result['candidate_miss_rate'] == .5
    assert result['n_known_candidate_rows'] == 3


def test_scaler_fit_on_training_only_and_scorer_contract(fixture):
    table = pd.read_parquet(fixture / 'candidates.parquet')
    train = table[table.trip_id != '00000005']
    validation = table[table.trip_id == '00000005'].copy()
    scorer, metadata = fit_model('knn', train, 0)
    assert np.allclose(scorer.preprocessor.mean_, train[FEATURES].mean().to_numpy(), rtol=1e-5, atol=1e-5)
    old_mean = scorer.preprocessor.mean_.copy()
    validation['distance_m'] += 10000
    probabilities = scorer.predict_proba(validation)
    assert probabilities.shape == (len(validation),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.array_equal(scorer.preprocessor.mean_, old_mean)
    assert metadata['n_fit_trips'] == 4


def test_invalid_ids_and_evaluator_partition_rejected(fixture):
    with pytest.raises(ValueError, match='restricted'):
        trip_directories(fixture, 'evaluator_test')
    folder = fixture / '00000001'
    (folder / '00000001.arcs').write_text('0 999\n')
    with pytest.raises(ValueError, match='out of range'):
        load_trip(folder)
