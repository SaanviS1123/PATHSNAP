"""One command for A1-A8, schema, folds, a synthetic fixture and EDA.

python -m src.data_processing.run_member1
Only the two explicit development partitions can be read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src.data_processing.loaders import load_trip, trip_directories
from src.data_processing.geo import project_trip
from src.data_processing.quality import inventory_rows, quality_row
from src.features.candidates import CandidateIndex
from src.features.labels import align_labels
from src.features.build import FEATURES, SCHEMA_VERSION, build_features

SEED = 42
RADII = [25, 50, 100]
CAPS = [4, 8, 16]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = Path('results/metrics')
    metrics.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / 'member1_cache'
    cache.mkdir(exist_ok=True)
    inventory, quality, alignment, sweeps, all_labels, hashes = [], [], [], [], [], []
    directories = {p: trip_directories(args.raw_dir, p) for p in ['train', 'student_test']}
    # Sweep TRAIN ONLY, before looking at student-test labels or results.
    for partition, folders in directories.items():
        for folder in folders:
            print(f'Prepare {partition}/{folder.name}', flush=True)
            trip = load_trip(folder, include_route=True)
            metric = project_trip(trip)
            inventory.extend(inventory_rows(folder, partition))
            quality.append(quality_row(metric, partition))
            labels = align_labels(metric)
            labels['partition'] = partition
            all_labels.append(labels)
            labels.to_parquet(cache / f'{partition}_{trip.trip_id}_labels.parquet', index=False)
            alignment.append(dict(partition=partition, trip_id=trip.trip_id, n_obs=len(labels),
                                  unknown_labels=int((~labels.label_known).sum()),
                                  median_distance_m=float(labels.alignment_distance_m.median()),
                                  p95_distance_m=float(labels.alignment_distance_m.quantile(.95)),
                                  max_distance_m=float(labels.alignment_distance_m.max()),
                                  skipped_route_occurrences=len(trip.route) - labels.route_position.nunique(),
                                  route_breaks=quality[-1]['route_breaks']))
            if partition == 'train':
                index = CandidateIndex(metric)
                for radius in RADII:
                    candidates = index.query(radius, max(CAPS))
                    for k in CAPS:
                        subset = candidates[candidates.candidate_rank <= k]
                        found = subset.merge(labels[['obs_idx', 'ref_arc']], on='obs_idx')
                        hits = found.loc[found.arc_id == found.ref_arc, 'obs_idx'].nunique()
                        known = int(labels.label_known.sum())
                        sweeps.append(dict(trip_id=trip.trip_id, radius_m=radius, k=k,
                                           known_obs=known, misses=known-hits, n_candidates=len(subset),
                                           n_obs=len(labels)))
            for suffix in ['track', 'nodes', 'arcs', 'route']:
                path = folder / f'{trip.trip_id}.{suffix}'
                hashes.append(dict(partition=partition, trip_id=trip.trip_id, file_type=suffix,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    sweep = pd.DataFrame(sweeps)
    sweep.to_csv(metrics / 'candidate_sweep_by_trip.csv', index=False)
    summary = sweep.groupby(['radius_m', 'k'], as_index=False)[['known_obs', 'misses', 'n_candidates', 'n_obs']].sum()
    summary['candidate_miss_rate'] = summary.misses / summary.known_obs
    summary['mean_candidates'] = summary.n_candidates / summary.n_obs
    summary.to_csv(metrics / 'candidate_sweep.csv', index=False)
    eligible = summary[summary.candidate_miss_rate <= .01]
    if len(eligible):
        chosen = eligible.sort_values(['mean_candidates', 'candidate_miss_rate', 'radius_m', 'k']).iloc[0]
    else:
        chosen = summary.sort_values(['candidate_miss_rate', 'mean_candidates', 'radius_m', 'k']).iloc[0]
    params = dict(radius_m=int(chosen.radius_m), k=int(chosen.k), selection_partition='train',
                  policy='Smallest mean candidate count with <=1% miss rate, otherwise lowest miss rate',
                  label_max_distance_m=100, schema_version=SCHEMA_VERSION, seed=SEED)
    write_json('configs/candidates.json', params)
    write_json('models/feature_schema.json', dict(version=SCHEMA_VERSION,
               features=[dict(name=x, dtype='float32') for x in FEATURES],
               identifiers=['trip_id', 'obs_idx', 'arc_id'], label='is_ref',
               unknown_label=-1, candidate_config=params))
    for partition, folders in directories.items():
        tables = []
        for folder in folders:
            print(f'Features {partition}/{folder.name}', flush=True)
            metric = project_trip(load_trip(folder))  # NO .route access
            table = build_features(metric, params['radius_m'], params['k'])
            labels = pd.read_parquet(cache / f'{partition}_{folder.name}_labels.parquet')
            table = table.merge(labels[['trip_id', 'obs_idx', 'ref_arc']], on=['trip_id', 'obs_idx'], validate='many_to_one')
            table['is_ref'] = np.where(table.ref_arc < 0, -1, (table.arc_id == table.ref_arc).astype(int)).astype('int8')
            table.drop(columns='ref_arc', inplace=True)
            tables.append(table)
        pd.concat(tables, ignore_index=True).to_parquet(args.output_dir / f'{partition}.parquet', index=False)
    pd.concat(all_labels, ignore_index=True).to_parquet(args.output_dir / 'obs_labels.parquet', index=False)
    for name, rows in [('inventory', inventory), ('quality_report', quality), ('label_alignment', alignment), ('data_manifest', hashes)]:
        pd.DataFrame(rows).to_csv(metrics / f'{name}.csv', index=False)
    # GroupKFold with explicit, reusable per-trip validation assignments.
    train = pd.read_parquet(args.output_dir / 'train.parquet', columns=['trip_id'])
    fold_map = {}
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, validation) in enumerate(splitter.split(train, groups=train.trip_id)):
        fold_map.update({trip_id: fold for trip_id in train.iloc[validation].trip_id.unique()})
    write_json('configs/folds.json', dict(method='GroupKFold', n_splits=5, shuffle=True, seed=SEED, trip_to_fold=fold_map))
    from src.data_processing.anomalies import diagnose
    from src.visualization.eda_diagnostics import plot_diagnostics
    diagnose(args.raw_dir)
    plot_diagnostics(args.raw_dir)
    from src.visualization.eda_maps import create_eda
    create_eda(pd.DataFrame(quality), args.raw_dir)
    from src.data_processing.synthetic import write_fixture
    write_fixture()
    write_json(metrics / 'member1_preparation.json', dict(elapsed_s=time.perf_counter()-start,
               seed=SEED, partitions={p: len(v) for p, v in directories.items()}, candidate_config=params))
    print('Member 1 preparation complete', flush=True)


if __name__ == '__main__':
    main()
