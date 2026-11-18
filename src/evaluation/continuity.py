"""Classifier-only directed continuity, pending adoption by Member 3's C1.

All consecutive observations count, including unknown labels and large gaps.
No shortest-path bridging is assumed: same arc or exact head-to-tail adjacency.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=100)
def _arcs(trip_id: str):
    if len(trip_id) != 8 or not trip_id.isdigit():
        raise ValueError('Expected an eight-digit trip ID')
    for partition in ['train', 'student_test']:
        path = ROOT / 'data/raw' / partition / trip_id / f'{trip_id}.arcs'
        if path.exists():
            return np.loadtxt(path, dtype=np.int64, ndmin=2)
    raise FileNotFoundError(f'No development road network for {trip_id}')


def route_continuity(predictions, networks=None):
    good, total = 0, 0
    for trip_id, group in predictions.groupby('trip_id', sort=False):
        group = group.sort_values('obs_idx')
        ids = group.arc_id.to_numpy(dtype=int)
        observations = group.obs_idx.to_numpy(dtype=int)
        if len(ids) < 2:
            continue
        arcs = networks[str(trip_id)] if networks is not None else _arcs(str(trip_id))
        if (ids < 0).any() or (ids >= len(arcs)).any():
            raise ValueError(f'Predicted arc outside network: {trip_id}')
        consecutive = np.diff(observations) == 1
        connected = (ids[:-1] == ids[1:]) | (arcs[ids[:-1], 1] == arcs[ids[1:], 0])
        good += int((consecutive & connected).sum())
        total += len(ids)-1
    return float(good/total) if total else None
