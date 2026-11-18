"""Additional real gap/connectivity/sweep plots, no model outputs needed."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pathsnap-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from src.data_processing.loaders import load_trip
from src.data_processing.geo import project_trip


def plot_diagnostics(raw_dir=Path('data/raw')):
    gaps = pd.read_csv('results/metrics/timestamp_gaps.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for partition, frame in gaps.groupby('partition'):
        ax.scatter(frame.time_gap_s, frame['count'], label=partition, alpha=.7)
    ax.set(xscale='log', yscale='log', xlabel='Time gap (s)', ylabel='Number of intervals', title='Actual timestamp interval distributions')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/plots/eda_timestamp_gaps.png', dpi=150)
    plt.close(fig)
    quality = pd.read_csv('results/metrics/quality_report.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for partition, frame in quality.groupby('partition'):
        ax.scatter(frame.n_arcs, frame.weak_components, label=partition, alpha=.7)
    ax.set(xscale='log', yscale='log', xlabel='Directed arcs', ylabel='Weakly connected components', title='Network connectivity by development partition')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/plots/eda_connectivity.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for partition, frame in quality.groupby('partition'):
        ax.scatter((frame.lon_min+frame.lon_max)/2, (frame.lat_min+frame.lat_max)/2, label=partition, alpha=.7)
    ax.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)', title='Development trip bounding-box centres')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/plots/eda_geography.png', dpi=150)
    plt.close(fig)
    alignment = pd.read_csv('results/metrics/label_alignment.csv', dtype={'trip_id': str})
    selected = alignment[alignment.partition == 'train'].sort_values('unknown_labels', ascending=False).iloc[0]
    metric = project_trip(load_trip(Path(raw_dir) / 'train' / selected.trip_id, include_route=True))
    centre = metric.xy[len(metric.xy)//2]
    lower, upper = centre-500, centre+500
    segments = metric.node_xy[metric.trip.arcs[metric.trip.route]]
    intersects = (segments.max(axis=1) >= lower).all(axis=1) & (segments.min(axis=1) <= upper).all(axis=1)
    in_view = ((metric.xy >= lower) & (metric.xy <= upper)).all(axis=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(LineCollection(segments[intersects], colors='#3d8295', linewidths=2, label='Supplied reference arcs'))
    ax.scatter(*metric.xy[in_view].T, color='#ce632f', s=5, label='Raw GPS')
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xlabel='Local easting (m)', ylabel='Local northing (m)', title=f'Train {selected.trip_id}: reference-distance outlier')
    ax.set_aspect('equal')
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/plots/eda_reference_offset.png', dpi=150)
    plt.close(fig)
    sweep = pd.read_csv('results/metrics/candidate_sweep.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    for radius, frame in sweep.groupby('radius_m'):
        ax.plot(frame.k, frame.candidate_miss_rate*100, marker='o', label=f'{radius} m')
    ax.axhline(1, color='grey', linestyle='--', label='1% selection threshold')
    ax.set(xlabel='Maximum candidates K', ylabel='Known-label candidate miss (%)', title='Training-only candidate search sweep', xticks=[4, 8, 16])
    ax.legend()
    fig.tight_layout()
    fig.savefig('results/plots/eda_candidate_sweep.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    plot_diagnostics()
