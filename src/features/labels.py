"""Offline approximate observation labels from ordered reference arc occurrences.

Dynamic programming permits staying on an occurrence or advancing, never reversing
route order. Repeated arc IDs remain separate states. No candidate set is consulted.
"""
import numpy as np
import pandas as pd
from src.data_processing.geo import MetricTrip, motion, segment_projection


def align_labels(metric: MetricTrip, max_distance_m: float = 100.0) -> pd.DataFrame:
    route = metric.trip.route
    if route is None or not len(route):
        raise ValueError('Offline alignment requires a nonempty reference route')
    n, m = len(metric.xy), len(route)
    unit, _, _, valid = motion(metric)
    vectors = metric.vector[route]
    lengths = metric.length[route]
    directions = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors), where=lengths[:, None] > 0)
    back = np.empty((n, m), dtype=np.int32)
    previous = np.zeros(m)
    indices = np.arange(m)
    # O(observations * route occurrences), no all-pairs road-network search.
    for i in range(n):
        distance, _ = segment_projection(metric.xy[i], metric.start[route], vectors)
        heading_cost = 15.0 * (1 - np.clip(directions @ unit[i], -1, 1)) if valid[i] else 0.0
        prefix = np.minimum.accumulate(previous)
        changed = np.r_[True, prefix[1:] < prefix[:-1]]
        parent = np.maximum.accumulate(np.where(changed, indices, 0))
        back[i] = parent
        previous = prefix + distance + heading_cost
    state = int(np.argmin(previous))
    states = np.empty(n, dtype=np.int32)
    for i in range(n - 1, -1, -1):
        states[i] = state
        state = back[i, state]
    arc_ids = route[states]
    distance, _ = segment_projection(metric.xy, metric.start[arc_ids], metric.vector[arc_ids])
    known = distance <= max_distance_m
    return pd.DataFrame({'trip_id': metric.trip.trip_id, 'obs_idx': np.arange(n),
                         'ref_arc': np.where(known, arc_ids, -1), 'route_position': states,
                         'alignment_distance_m': distance, 'label_known': known})
