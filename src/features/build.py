"""Inference-safe candidate feature extraction. No reference route access."""
import numpy as np
from src.data_processing.geo import MetricTrip, motion
from src.features.candidates import CandidateIndex

FEATURES = ['distance_m', 'position_frac', 'heading_diff_rad', 'heading_valid',
            'speed_mps', 'arc_length_m', 'candidate_rank', 'in_degree', 'out_degree',
            'time_gap_s', 'large_gap', 'outside_radius']
SCHEMA_VERSION = 'member1-v1'


def build_features(metric: MetricTrip, radius_m: float, k: int, index=None):
    table = (index or CandidateIndex(metric)).query(radius_m, k)
    obs, arc = table.obs_idx.to_numpy(), table.arc_id.to_numpy()
    unit, dt, speed, valid = motion(metric)
    direction = np.divide(metric.vector[arc], metric.length[arc, None],
                          out=np.zeros((len(arc), 2)), where=metric.length[arc, None] > 0)
    angle = np.arccos(np.clip(np.sum(direction * unit[obs], axis=1), -1, 1))
    table['heading_diff_rad'] = np.where(valid[obs], angle, 0)
    table['heading_valid'] = valid[obs].astype(float)
    table['speed_mps'] = speed[obs]
    table['arc_length_m'] = metric.length[arc]
    incoming = np.bincount(metric.trip.arcs[:, 1], minlength=len(metric.node_xy))
    outgoing = np.bincount(metric.trip.arcs[:, 0], minlength=len(metric.node_xy))
    table['in_degree'] = incoming[metric.trip.arcs[arc, 0]]
    table['out_degree'] = outgoing[metric.trip.arcs[arc, 1]]
    table['time_gap_s'] = dt[obs]
    table['large_gap'] = ((dt[obs] > 5) | (dt[obs] <= 0)).astype(float)
    table[FEATURES] = table[FEATURES].astype('float32')
    return table[['trip_id', 'obs_idx', 'arc_id', *FEATURES]]
