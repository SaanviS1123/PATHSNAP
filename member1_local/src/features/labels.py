
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data_processing.loaders import load_trip
from src.data_processing.geo import LocalMetricProjection


def align_route_to_gps(
    track,
    nodes,
    arcs,
    route,
    k=30,
    backward_penalty=2.0
):
    """
    Produce provisional per-observation arc labels.

    Uses spatial proximity to reference-route segments
    and penalizes backward movement in route order.

    These are heuristic labels, not independently
    verified ground truth.
    """

    projection = LocalMetricProjection(nodes)

    gps_xy = projection.transform(track[:, :2])
    node_xy = projection.transform(nodes)

    route = np.asarray(route, dtype=int)

    starts = node_xy[arcs[route, 0]]
    ends = node_xy[arcs[route, 1]]

    vectors = ends - starts
    lengths_sq = np.sum(vectors ** 2, axis=1)

    midpoints = (starts + ends) / 2

    tree = cKDTree(midpoints)

    records = []
    previous_position = 0

    for obs_id, point in enumerate(gps_xy):
        neighbor_count = min(k, len(route))

        _, positions = tree.query(
            point,
            k=neighbor_count
        )

        positions = np.atleast_1d(
            positions
        ).astype(int)

        a = starts[positions]
        v = vectors[positions]

        fractions = np.clip(
            np.sum((point - a) * v, axis=1)
            / np.maximum(
                lengths_sq[positions],
                1e-12
            ),
            0,
            1
        )

        closest_points = (
            a + fractions[:, None] * v
        )

        distances = np.linalg.norm(
            closest_points - point,
            axis=1
        )

        backward_steps = np.maximum(
            0,
            previous_position - positions
        )

        scores = (
            distances
            + backward_penalty * backward_steps
        )

        best_index = int(np.argmin(scores))
        best_position = int(positions[best_index])

        records.append({
            "observation_id": obs_id,
            "arc_id": int(route[best_position]),
            "route_position": best_position,
            "distance_m": float(
                distances[best_index]
            ),
            "backward_jump": bool(
                best_position < previous_position
            )
        })

        previous_position = best_position

    return pd.DataFrame(records)


def main():
    folder = (
        Path("Student folder")
        / "train"
        / "00000001"
    )

    trip = load_trip(folder)

    labels = align_route_to_gps(
        trip["track"],
        trip["nodes"],
        trip["arcs"],
        trip["route"]
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels.insert(
        0,
        "trip_id",
        trip["trip_id"]
    )

    output_path = (
        output_dir / "obs_labels_trip1.csv"
    )

    labels.to_csv(
        output_path,
        index=False
    )

    print("A4 LABEL ALIGNMENT TEST")
    print("Trip:", trip["trip_id"])
    print("GPS observations:", len(labels))

    print(
        "Unique assigned arcs:",
        labels["arc_id"].nunique()
    )

    print(
        "Median snap distance (m):",
        round(
            labels["distance_m"].median(),
            2
        )
    )

    print(
        "95th percentile distance (m):",
        round(
            labels["distance_m"].quantile(0.95),
            2
        )
    )

    print(
        "Backward jumps:",
        int(labels["backward_jump"].sum())
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
