"""Strict loaders. Arc/node IDs are zero-based file row numbers, local to a trip."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None = None


def _read(path: Path, columns: int, integer: bool = False) -> np.ndarray:
    values = np.loadtxt(path, ndmin=2)
    if not values.size or values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError(f'{path}: expected nonempty, finite {columns}-column data')
    if integer:
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f'{path}: IDs must be integers')
        return values.astype(np.int64)
    return values


def load_trip(trip_dir: str | Path, *, include_route: bool = False) -> Trip:
    """Default is inference-safe: never stat/open the reference route."""
    folder = Path(trip_dir)
    trip_id = folder.name
    if len(trip_id) != 8 or not trip_id.isdigit():
        raise ValueError(f'Expected an eight-digit trip directory: {folder}')
    track = _read(folder / f'{trip_id}.track', 3)
    nodes = _read(folder / f'{trip_id}.nodes', 2)
    arcs = _read(folder / f'{trip_id}.arcs', 2, True)
    for name, coordinates in [('track', track[:, :2]), ('nodes', nodes)]:
        if (np.abs(coordinates[:, 0]) > 180).any() or (np.abs(coordinates[:, 1]) > 90).any():
            raise ValueError(f'{folder}: invalid lon/lat in {name}')
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError(f'{folder}: arc node ID out of range')
    route = None
    if include_route:
        route = _read(folder / f'{trip_id}.route', 1, True).ravel()
        if (route < 0).any() or (route >= len(arcs)).any():
            raise ValueError(f'{folder}: route arc ID out of range')
    return Trip(trip_id, track, nodes, arcs, route)


def trip_directories(raw_dir: str | Path, partition: str) -> list[Path]:
    if partition not in {'train', 'student_test'}:
        raise ValueError('Development is restricted to train and student_test')
    folders = sorted(p for p in (Path(raw_dir) / partition).iterdir() if p.is_dir())
    if not folders:
        raise ValueError(f'No trips in {partition}')
    return folders
