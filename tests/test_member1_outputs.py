"""Validation of real regenerated development outputs (skip on a clean clone)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.features.build import FEATURES

pytestmark = pytest.mark.skipif(not Path('data/processed/train.parquet').exists(), reason='Raw data/preparation not available')


def test_full_tables_and_group_folds():
    folds = json.loads(Path('configs/folds.json').read_text())['trip_to_fold']
    labels = pd.read_parquet('data/processed/obs_labels.parquet')
    assert len(folds) == 80
    assert set(folds.values()) == set(range(5))
    train_trips, student_trips = set(), set()
    for partition, expected in [('train', 80), ('student_test', 10)]:
        table = pd.read_parquet(f'data/processed/{partition}.parquet')
        reference = labels[labels.partition == partition]
        assert table.trip_id.nunique() == expected
        assert not table.duplicated(['trip_id', 'obs_idx', 'arc_id']).any()
        assert np.isfinite(table[FEATURES]).all().all()
        assert all(str(dtype) == 'float32' for dtype in table[FEATURES].dtypes)
        assert table.groupby(['trip_id', 'obs_idx']).size().max() <= 16
        covered = table[['trip_id', 'obs_idx']].drop_duplicates()
        assert len(covered) == len(reference)
        joined = table.merge(reference[['trip_id', 'obs_idx', 'ref_arc']], on=['trip_id', 'obs_idx'], validate='many_to_one')
        assert joined.loc[joined.ref_arc < 0, 'is_ref'].eq(-1).all()
        known = joined.ref_arc >= 0
        assert np.array_equal(joined.loc[known, 'is_ref'], (joined.loc[known, 'arc_id'] == joined.loc[known, 'ref_arc']).astype(int))
        assert table.groupby(['trip_id', 'obs_idx']).is_ref.apply(lambda x: (x == 1).sum()).max() <= 1
        if partition == 'train':
            train_trips = set(table.trip_id)
        else:
            student_trips = set(table.trip_id)
    assert not train_trips & student_trips
    assert train_trips == set(folds)


def test_candidate_sweep_is_train_only_and_matches_export():
    folds = json.loads(Path('configs/folds.json').read_text())['trip_to_fold']
    sweep = pd.read_csv('results/metrics/candidate_sweep_by_trip.csv', dtype={'trip_id': str})
    assert set(sweep.trip_id) == set(folds)
    config = json.loads(Path('configs/candidates.json').read_text())
    summary = pd.read_csv('results/metrics/candidate_sweep.csv')
    choice = summary[(summary.radius_m == config['radius_m']) & (summary.k == config['k'])].iloc[0]
    train = pd.read_parquet('data/processed/train.parquet')
    labels = pd.read_parquet('data/processed/obs_labels.parquet')
    known = labels[(labels.partition == 'train') & (labels.ref_arc >= 0)]
    hits = train[train.is_ref == 1]
    assert choice.candidate_miss_rate == pytest.approx(1-len(hits)/len(known))
