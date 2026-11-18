"""Classifier-only demo for one development trip: score it, print the numbers, write a Folium map.

Run through scripts/demo.sh. Inference reads only .track/.nodes/.arcs. The aligned reference
labels (data/processed/obs_labels.parquet) are used afterwards, offline, to report agreement;
they are never a model input. No HMM or Viterbi decoding is involved: each GPS point simply
takes its highest-scoring candidate arc.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd

from src.data_processing.geo import project_trip
from src.data_processing.loaders import load_trip
from src.features.build import build_features

ROOT = Path(__file__).resolve().parents[1]
GPS, MODEL, NEAREST, REFERENCE, NETWORK = '#d65128', '#1f6fb4', '#7b3fa0', '#2a9d4a', '#657785'


def show_comparison():
    """Print whichever measured comparison table exists (never invents a number)."""
    metrics = ROOT / 'results/metrics'
    for name, wanted in [('model_comparison.csv', ['model_name', 'cv_top1_accuracy', 'cv_macro_f1', 'cv_log_loss',
                                                    'student_top1_accuracy', 'student_route_continuity']),
                         ('member1_model_comparison.csv', ['model', 'cv_macro_f1', 'student_top1', 'log_loss',
                                                            'inference_time_s'])]:
        path = metrics / name
        if path.exists():
            table = pd.read_csv(path)
            table = table[[c for c in wanted if c in table.columns]]
            print(f'\nModel comparison ({path.relative_to(ROOT)})')
            print(table.round(4).to_string(index=False))
            return
    print('\nNo model comparison table yet: run scripts/train.sh baselines')


def load_any_scorer(folder: Path):
    """Member 2 exports (feature_schema_version) and Member 1 exports (schema_version) differ."""
    card = json.loads((folder / 'model_card.json').read_text())
    if 'feature_schema_version' in card:
        from src.models.training import load_scorer
        return load_scorer(folder), card.get('model_name', folder.name), card.get('status', '')
    from src.models.member1_harness import load_scorer
    return load_scorer(folder), card.get('name', folder.name), 'member1'


def find_trip(trip_id: str) -> Path:
    for partition in ['student_test', 'train']:
        folder = ROOT / 'data/raw' / partition / trip_id
        if folder.is_dir():
            return folder
    sys.exit(f'error: no trip {trip_id} under data/raw/{{student_test,train}}')


def default_trip(labels: pd.DataFrame) -> str:
    sizes = labels[labels.partition == 'student_test'].groupby('trip_id').size()
    return str(sizes.idxmin())


def segments(trip, arc_ids):
    """GeoJSON MultiLineString for a set of arcs (lon/lat order)."""
    arcs = trip.arcs[np.unique(arc_ids)]
    return {'type': 'Feature', 'properties': {},
            'geometry': {'type': 'MultiLineString', 'coordinates': trip.nodes[arcs].tolist()}}


def add_lines(map_, name, trip, arc_ids, color, weight, show=True, dash=None):
    group = folium.FeatureGroup(name=name, show=show)
    style = {'color': color, 'weight': weight, 'opacity': .9, 'dashArray': dash}
    folium.GeoJson(segments(trip, arc_ids), style_function=lambda _: style).add_to(group)
    group.add_to(map_)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trip')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--with-reference', action='store_true')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()

    show_comparison()
    labels = pd.read_parquet(ROOT / 'data/processed/obs_labels.parquet')
    trip_id = args.trip or default_trip(labels)
    folder = find_trip(trip_id)
    model_dir = args.model_dir or (ROOT / 'models' if (ROOT / 'models/model_card.json').exists()
                                   else ROOT / 'models/member1/mlp')
    scorer, model_name, status = load_any_scorer(model_dir)

    trip = load_trip(folder)  # no .route: inference path
    metric = project_trip(trip)
    config = json.loads((ROOT / 'configs/candidates.json').read_text())
    candidates = build_features(metric, config['radius_m'], config['k'])
    candidates['score'] = scorer.predict_proba(candidates)
    ranked = candidates.sort_values(['obs_idx', 'score', 'arc_id'], ascending=[True, False, True])
    chosen = ranked.drop_duplicates('obs_idx')
    nearest = candidates[candidates.candidate_rank == 1].drop_duplicates('obs_idx')

    n_obs = len(trip.track)
    print(f'\nTrip {trip_id}: {n_obs} GPS points, {len(trip.arcs)} road arcs, '
          f'{len(candidates) / n_obs:.1f} candidates per point')
    print(f'Scorer: {model_name} ({status or "exported"}) from {model_dir.relative_to(ROOT)}')
    known = labels[(labels.trip_id == trip_id) & (labels.ref_arc >= 0)]
    if len(known):
        def agree(picks):
            joined = known.merge(picks[['obs_idx', 'arc_id']], on='obs_idx', how='left')
            return float((joined.arc_id == joined.ref_arc).mean())
        print(f'Agreement with the aligned reference (offline, {len(known)} labelled points; '
              f'the alignment is approximate supervision, not point-level ground truth):')
        print(f'  classifier top-1 {agree(chosen):.3f}   nearest-arc baseline {agree(nearest):.3f}')
    try:
        from src.evaluation.continuity import route_continuity
        continuity = route_continuity(chosen[['trip_id', 'obs_idx', 'arc_id']])
        if continuity is not None:
            print(f'Classifier route continuity (same or head-to-tail arcs between consecutive points): {continuity:.3f}')
    except Exception as error:  # continuity is Member 2's interim metric; the demo must not hinge on it
        print(f'(continuity skipped: {error})')

    lonlat = trip.track[:, :2][:, ::-1]
    map_ = folium.Map(location=lonlat.mean(axis=0).tolist(), zoom_start=14, tiles='OpenStreetMap')
    # Road network cropped to the GPS bounding box plus ~500 m, so the page stays small.
    low, high = lonlat.min(axis=0) - .005, lonlat.max(axis=0) + .005
    midpoints = trip.nodes[trip.arcs].mean(axis=1)[:, ::-1]
    near = np.flatnonzero(((midpoints >= low) & (midpoints <= high)).all(axis=1))
    add_lines(map_, 'Road network (cropped)', trip, near, NETWORK, 1, show=False)
    add_lines(map_, f'Classifier route ({model_name})', trip, chosen.arc_id.to_numpy(), MODEL, 5)
    add_lines(map_, 'Nearest-arc baseline', trip, nearest.arc_id.to_numpy(), NEAREST, 3, show=False, dash='6 6')
    if args.with_reference:
        reference = load_trip(folder, include_route=True).route  # offline report layer only
        add_lines(map_, 'Reference route (offline)', trip, reference, REFERENCE, 3, dash='2 8')
    gps = folium.FeatureGroup(name='Raw GPS')
    folium.PolyLine(lonlat.tolist(), color=GPS, weight=2, opacity=.6).add_to(gps)
    for index in range(0, n_obs, max(1, n_obs // 800)):  # at most ~800 markers
        folium.CircleMarker(lonlat[index].tolist(), radius=2, color=GPS,
                            tooltip=f'obs {index}, t={trip.track[index, 2]:g}s').add_to(gps)
    gps.add_to(map_)
    folium.LayerControl().add_to(map_)
    map_.fit_bounds([low.tolist(), high.tolist()])
    legend = (f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:10px;'
              f'font:13px sans-serif">Trip {trip_id} · <b style="color:{GPS}">GPS</b> · '
              f'<b style="color:{MODEL}">classifier</b> · <b style="color:{NEAREST}">nearest arc</b>'
              + (f' · <b style="color:{REFERENCE}">reference</b>' if args.with_reference else '') + '</div>')
    map_.get_root().html.add_child(folium.Element(legend))
    out = ROOT / 'maps' / f'demo_{trip_id}.html'
    out.parent.mkdir(exist_ok=True)
    map_.save(out)
    print(f'\nMap written to {out.relative_to(ROOT)}')
    if not args.no_open:
        subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', str(out)], check=False)


if __name__ == '__main__':
    main()
