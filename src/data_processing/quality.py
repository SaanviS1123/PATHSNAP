"""Reproducible partition-labelled inventory, quality and directed graph statistics."""
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.data_processing.geo import MetricTrip


def inventory_rows(folder: Path, partition: str):
    rows = []
    for suffix, columns in [('track', 3), ('nodes', 2), ('arcs', 2), ('route', 1)]:
        path = folder / f'{folder.name}.{suffix}'
        values = np.loadtxt(path, ndmin=2)
        rows.append(dict(partition=partition, trip_id=folder.name, file_type=suffix,
                         rows=len(values), columns=columns, bytes=path.stat().st_size,
                         dtype='int64' if suffix in {'arcs', 'route'} else 'float64'))
    return rows


def quality_row(metric: MetricTrip, partition: str):
    trip = metric.trip
    graph = coo_matrix((np.ones(len(trip.arcs)), trip.arcs.T),
                       shape=(len(trip.nodes), len(trip.nodes))).tocsr()
    dt = np.diff(trip.track[:, 2])
    degree = np.bincount(trip.arcs.ravel(), minlength=len(trip.nodes))
    pairs = set(map(tuple, trip.arcs))
    route = trip.route
    breaks = int(np.sum(trip.arcs[route[:-1], 1] != trip.arcs[route[1:], 0]))
    return dict(partition=partition, trip_id=trip.trip_id, n_obs=len(trip.track),
                n_nodes=len(trip.nodes), n_arcs=len(trip.arcs), n_route_arcs=len(route),
                missing_values=int(np.isnan(trip.track).sum()), invalid_coordinates=0,
                duplicate_timestamps=int((dt == 0).sum()), reversed_timestamps=int((dt < 0).sum()),
                irregular_gaps=int((~np.isclose(dt, 1)).sum()), large_gaps=int((dt > 5).sum()),
                max_gap_s=float(dt.max()) if len(dt) else 0,
                lon_min=trip.track[:, 0].min(), lon_max=trip.track[:, 0].max(),
                lat_min=trip.track[:, 1].min(), lat_max=trip.track[:, 1].max(),
                weak_components=connected_components(graph, directed=True, connection='weak')[0],
                strong_components=connected_components(graph, directed=True, connection='strong')[0],
                isolated_nodes=int((degree == 0).sum()), zero_length_arcs=int((metric.length == 0).sum()),
                reverse_arc_share=sum((b, a) in pairs for a, b in trip.arcs) / len(trip.arcs),
                route_breaks=breaks, metric_crs=metric.crs)
