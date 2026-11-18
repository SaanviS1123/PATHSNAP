
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.loaders import load_trip
from src.data_processing.geo import LocalMetricProjection
from src.features.candidates import RoadCandidateIndex

TRIP_IDS = ["00000033", "00000054"]
SEARCH_K_VALUES = [500, 2000]
CANDIDATE_K = 30
RADIUS_M = 100


def true_arc_distance(index, point, arc_id):
    start = index.starts[arc_id]
    vector = index.vectors[arc_id]
    length_sq = index.lengths_sq[arc_id]

    fraction = np.clip(
        np.dot(point - start, vector)
        / max(length_sq, 1e-12),
        0,
        1
    )

    closest = start + fraction * vector
    return float(np.linalg.norm(point - closest))


def main():
    labels = pd.read_parquet(
        "data/processed/obs_labels.parquet"
    )
    labels["trip_id"] = (
        labels["trip_id"].astype(str).str.zfill(8)
    )

    print("CANDIDATE FAILURE DIAGNOSIS")

    for trip_id in TRIP_IDS:
        folder = Path("Student folder") / "train" / trip_id
        trip = load_trip(folder)

        projection = LocalMetricProjection(trip["nodes"])
        gps_xy = projection.transform(trip["track"][:, :2])
        index = RoadCandidateIndex(
            trip["nodes"],
            trip["arcs"],
            projection
        )

        trip_labels = labels[
            labels["trip_id"] == trip_id
        ].sort_values("observation_id")

        label_ids = trip_labels["arc_id"].to_numpy(
            dtype=int
        )

        if len(label_ids) != len(gps_xy):
            raise ValueError(
                f"Label mismatch for {trip_id}"
            )

        counts = {
            search_k: 0
            for search_k in SEARCH_K_VALUES
        }
        near_reference = 0
        far_reference = 0
        distances_to_reference = []

        for obs_id, point in enumerate(gps_xy):
            true_arc = int(label_ids[obs_id])

            reference_distance = true_arc_distance(
                index,
                point,
                true_arc
            )

            distances_to_reference.append(
                reference_distance
            )

            if reference_distance <= RADIUS_M:
                near_reference += 1
            else:
                far_reference += 1

            for search_k in SEARCH_K_VALUES:
                ids, distances = index.query(
                    point,
                    k=CANDIDATE_K,
                    search_k=search_k
                )

                valid_ids = ids[
                    distances <= RADIUS_M
                ]

                if true_arc in valid_ids:
                    counts[search_k] += 1

        n = len(gps_xy)

        print(f"\nTrip {trip_id}")
        print("GPS observations:", n)
        print(
            "Reference arc within 100 m:",
            f"{near_reference / n:.2%}"
        )
        print(
            "Reference arc beyond 100 m:",
            f"{far_reference / n:.2%}"
        )
        print(
            "Median reference distance:",
            f"{np.median(distances_to_reference):.1f} m"
        )
        print(
            "95th percentile reference distance:",
            f"{np.percentile(distances_to_reference, 95):.1f} m"
        )

        for search_k in SEARCH_K_VALUES:
            print(
                f"Coverage search_k={search_k}:",
                f"{counts[search_k] / n:.2%}"
            )


if __name__ == "__main__":
    main()
