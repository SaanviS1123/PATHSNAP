"""Additional real gap/connectivity/sweep plots, no model outputs needed."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pathsnap-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_diagnostics():
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
