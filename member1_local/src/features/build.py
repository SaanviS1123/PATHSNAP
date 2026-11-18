
import numpy as np
import pandas as pd

from src.data_processing.geo import LocalMetricProjection
from src.features.candidates import RoadCandidateIndex

FEATURE_COLUMNS = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]


def build_trip_features(
    trip,
    labels=None,
    search_k=500,
    candidate_k=30,
    radius_m=100
):
    """
    Generate candidate-level features for one trip.

    labels: optional DataFrame containing
            observation_id and arc_id.

    Never use .route during inference.
    """
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    projection = LocalMetricProjection(nodes)

    gps_xy = projection.transform(
        track[:, :2]
    )

    node_xy = projection.transform(nodes)

    candidate_index = RoadCandidateIndex(
        nodes,
        arcs,
        projection
    )

    road_vectors = (
        node_xy[arcs[:, 1]]
        - node_xy[arcs[:, 0]]
    )

    road_lengths = np.linalg.norm(
        road_vectors,
        axis=1
    )

    road_bearings = np.degrees(
        np.arctan2(
            road_vectors[:, 0],
            road_vectors[:, 1]
        )
    )

    label_lookup = {}

    if labels is not None:
        label_lookup = dict(
            zip(
                labels["observation_id"].astype(int),
                labels["arc_id"].astype(int)
            )
        )

    rows = []

    for obs_id, point in enumerate(gps_xy):
        candidate_ids, distances = (
            candidate_index.query(
                point,
                k=candidate_k,
                search_k=search_k
            )
        )

        within_radius = distances <= radius_m

        if np.any(within_radius):
            candidate_ids = candidate_ids[
                within_radius
            ]
            distances = distances[
                within_radius
            ]
        else:
            # Always retain the nearest candidate.
            candidate_ids = candidate_ids[:1]
            distances = distances[:1]

        if obs_id == 0:
            dt = 0.0
            speed = 0.0
            gps_bearing = 0.0
        else:
            displacement = (
                gps_xy[obs_id]
                - gps_xy[obs_id - 1]
            )

            dt = max(
                0.0,
                float(
                    track[obs_id, 2]
                    - track[obs_id - 1, 2]
                )
            )

            speed = (
                float(np.linalg.norm(displacement) / dt)
                if dt > 0 else 0.0
            )

            gps_bearing = np.degrees(
                np.arctan2(
                    displacement[0],
                    displacement[1]
                )
            )

        for arc_id, distance in zip(
            candidate_ids,
            distances
        ):
            arc_id = int(arc_id)

            difference = abs(
                (gps_bearing - road_bearings[arc_id] + 180)
                % 360 - 180
            )

            row = {
                "trip_id": trip["trip_id"],
                "observation_id": obs_id,
                "arc_id": arc_id,
                "distance_m": float(distance),
                "segment_length_m": float(
                    road_lengths[arc_id]
                ),
                "bearing_difference": float(
                    difference
                ),
                "speed_mps": speed,
                "time_delta_s": dt
            }

            if labels is not None:
                row["is_correct_arc"] = int(
                    arc_id == label_lookup.get(
                        obs_id, -1
                    )
                )

            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    from src.data_processing.loaders import load_trip

    folder = (
        Path("Student folder")
        / "train"
        / "00000001"
    )

    trip = load_trip(folder)

    labels = pd.read_parquet(
        "data/processed/obs_labels.parquet"
    )

    trip_labels = labels[
        labels["trip_id"].astype(str).str.zfill(8)
        == trip["trip_id"]
    ]

    features = build_trip_features(
        trip,
        labels=trip_labels
    )

    print("A6 FEATURE BUILDER TEST")
    print("Trip:", trip["trip_id"])
    print("Candidate rows:", len(features))
    print("GPS observations:", features["observation_id"].nunique())

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nFirst three rows:")
    print(features.head(3).to_string(index=False))

    print("\nPositive labels:", int(
        features["is_correct_arc"].sum()
    ))

    print("\nA6 FEATURE BUILDER WORKING")
