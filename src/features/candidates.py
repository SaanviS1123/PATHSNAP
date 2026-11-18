"""Exact segment-distance candidates using an STRtree; deterministic tie breaking."""
import numpy as np
import pandas as pd
from shapely import STRtree, linestrings, points
from src.data_processing.geo import MetricTrip, segment_projection


class CandidateIndex:
    def __init__(self, metric: MetricTrip):
        self.metric = metric
        self.tree = STRtree(linestrings(np.stack([metric.start, metric.start + metric.vector], axis=1)))

    def query(self, radius_m: float, k: int) -> pd.DataFrame:
        if radius_m <= 0 or k < 1:
            raise ValueError('radius_m and k must be positive')
        metric = self.metric
        geometries = points(metric.xy)
        obs, arcs = self.tree.query(geometries, predicate='dwithin', distance=radius_m)
        # A missed search still yields one inference candidate; never insert a label.
        missing = np.setdiff1d(np.arange(len(geometries)), np.unique(obs))
        if len(missing):
            nearest_obs, nearest_arc = self.tree.query_nearest(geometries[missing], all_matches=True)
            obs = np.concatenate([obs, missing[nearest_obs]])
            arcs = np.concatenate([arcs, nearest_arc])
        distance, fraction = segment_projection(metric.xy[obs], metric.start[arcs], metric.vector[arcs])
        order = np.lexsort((arcs, distance, obs))
        obs, arcs, distance, fraction = (x[order] for x in (obs, arcs, distance, fraction))
        starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(obs) != 0], np.arange(len(obs)), 0))
        rank = np.arange(len(obs)) - starts + 1
        keep = rank <= k
        return pd.DataFrame({'trip_id': metric.trip.trip_id, 'obs_idx': obs[keep],
                             'arc_id': arcs[keep], 'distance_m': distance[keep],
                             'position_frac': fraction[keep], 'candidate_rank': rank[keep],
                             'outside_radius': (distance[keep] > radius_m).astype(float)})
