"""Independent nearest-reference and geodesic diagnostics; offline only."""
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely import STRtree, linestrings, points, distance
from src.data_processing.geo import project_trip
from src.data_processing.loaders import load_trip, trip_directories


def diagnose(raw_dir=Path('data/raw')):
    rows, gaps = [], []
    geod = Geod(ellps='WGS84')
    for partition in ['train', 'student_test']:
        all_gaps = []
        for folder in trip_directories(raw_dir, partition):
            metric = project_trip(load_trip(folder, include_route=True))
            route = metric.trip.route
            segments = linestrings(np.stack([metric.start[route], metric.start[route]+metric.vector[route]], axis=1))
            observations = points(metric.xy)
            nearest = STRtree(segments).nearest(observations)
            distances = distance(observations, segments[nearest])
            sample = np.linspace(0, len(metric.trip.arcs)-1, min(1000, len(metric.trip.arcs)), dtype=int)
            arc_nodes = metric.trip.nodes[metric.trip.arcs[sample]]
            _, _, geodesic = geod.inv(arc_nodes[:, 0, 0], arc_nodes[:, 0, 1], arc_nodes[:, 1, 0], arc_nodes[:, 1, 1])
            relative = np.abs(metric.length[sample]-geodesic)/np.maximum(geodesic, 1e-9)
            rows.append(dict(partition=partition, trip_id=folder.name,
                             nearest_reference_median_m=float(np.median(distances)),
                             nearest_reference_p95_m=float(np.quantile(distances, .95)),
                             nearest_reference_over100m=int((distances > 100).sum()),
                             sampled_arcs=len(sample), max_projection_relative_error=float(relative.max())))
            all_gaps.extend(np.diff(metric.trip.track[:, 2]))
        values, counts = np.unique(all_gaps, return_counts=True)
        gaps.extend(dict(partition=partition, time_gap_s=float(value), count=int(count)) for value, count in zip(values, counts))
    pd.DataFrame(rows).to_csv('results/metrics/reference_distance_diagnostics.csv', index=False)
    pd.DataFrame(gaps).to_csv('results/metrics/timestamp_gaps.csv', index=False)


if __name__ == '__main__':
    diagnose()
