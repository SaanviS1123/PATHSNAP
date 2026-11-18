
from pathlib import Path

import numpy as np


def load_track(path):
    """Load GPS observations: longitude, latitude, timestamp."""
    data = np.loadtxt(path, ndmin=2)

    if data.shape[1] != 3:
        raise ValueError(
            f"Track must have 3 columns; got {data.shape[1]}"
        )

    if not np.isfinite(data).all():
        raise ValueError("Track contains NaN or infinite values")

    lon, lat = data[:, 0], data[:, 1]

    if np.any((lon < -180) | (lon > 180)):
        raise ValueError("Invalid longitude values")

    if np.any((lat < -90) | (lat > 90)):
        raise ValueError("Invalid latitude values")

    return data


def load_nodes(path):
    """Load road-network nodes: longitude, latitude."""
    data = np.loadtxt(path, ndmin=2)

    if data.shape[1] != 2:
        raise ValueError(
            f"Nodes must have 2 columns; got {data.shape[1]}"
        )

    if not np.isfinite(data).all():
        raise ValueError("Nodes contain NaN or infinite values")

    return data


def load_arcs(path, node_count=None):
    """Load directed arcs: start node ID, end node ID."""
    data = np.loadtxt(path, dtype=np.int64, ndmin=2)

    if data.shape[1] != 2:
        raise ValueError(
            f"Arcs must have 2 columns; got {data.shape[1]}"
        )

    if node_count is not None:
        if np.any(data < 0) or np.any(data >= node_count):
            raise ValueError("Arc references invalid node IDs")

    return data


def load_route(path, arc_count=None):
    """Load ordered reference-route arc IDs."""
    data = np.loadtxt(path, dtype=np.int64, ndmin=1)

    if arc_count is not None:
        if np.any(data < 0) or np.any(data >= arc_count):
            raise ValueError("Route references invalid arc IDs")

    return data


def load_trip(folder):
    """Load and validate all four files for one trip."""
    folder = Path(folder)
    trip_id = folder.name

    track = load_track(folder / f"{trip_id}.track")
    nodes = load_nodes(folder / f"{trip_id}.nodes")

    arcs = load_arcs(
        folder / f"{trip_id}.arcs",
        node_count=len(nodes)
    )

    route = load_route(
        folder / f"{trip_id}.route",
        arc_count=len(arcs)
    )

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route
    }


if __name__ == "__main__":
    folder = (
        Path("Student folder")
        / "train"
        / "00000001"
    )

    trip = load_trip(folder)

    print("A1 DATA LOADER TEST")
    print("Trip:", trip["trip_id"])
    print("GPS points:", len(trip["track"]))
    print("Nodes:", len(trip["nodes"]))
    print("Arcs:", len(trip["arcs"]))
    print("Reference route arcs:", len(trip["route"]))

    print("\nFirst GPS observation:")
    print(trip["track"][0])

    print("\nFirst road arc:")
    print(trip["arcs"][0])

    print("\nA1 LOADERS WORKING SUCCESSFULLY")
