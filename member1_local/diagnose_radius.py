
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.loaders import load_trip
from src.data_processing.geo import LocalMetricProjection
from src.features.candidates import RoadCandidateIndex

TRIP_IDS = ["00000033", "00000054"]
K_VALUES = [30, 50, 100]
RADII = [100, 150, 200, 250]
SEARCH_K = 2000
SAMPLE_PER_TRIP = 1000


def main():
    labels = pd.read_parquet(
        "data/processed/obs_labels.parquet"
    )
    labels["trip_id"] = (
        labels["trip_id"].astype(str).str.zfill(8)
    )

    print("TARGETED CANDIDATE SWEEP")

    for trip_id in TRIP_IDS:
        trip = load_trip(
            Path("Student folder") / "train" / trip_id
        )

        projection = LocalMetricProjection(
            trip["nodes"]
        )
        gps_xy = projection.transform(
            trip["track"][:, :2]
        )

        index = RoadCandidateIndex(
            trip["nodes"],
            trip["arcs"],
            projection
        )

        trip_labels = (
            labels[labels["trip_id"] == trip_id]
            .set_index("observation_id")
        )

        sample_ids = np.linspace(
            0,
            len(gps_xy) - 1,
            min(SAMPLE_PER_TRIP, len(gps_xy)),
            dtype=int
        )

        results = {
            (k, radius): 0
            for k in K_VALUES
            for radius in RADII
        }

        for obs_id in sample_ids:
            point = gps_xy[obs_id]
            reference_arc = int(
                trip_labels.loc[obs_id, "arc_id"]
            )

            ids, distances = index.query(
                point,
                k=max(K_VALUES),
                search_k=SEARCH_K
            )

            for k in K_VALUES:
                for radius in RADII:
                    selected = ids[:k][
                        distances[:k] <= radius
                    ]

                    if reference_arc in selected:
                        results[(k, radius)] += 1

        print(f"\nTrip {trip_id}")
        print("Sampled observations:", len(sample_ids))

        for k in K_VALUES:
            values = [
                f"{results[(k, radius)] / len(sample_ids):.2%}"
                for radius in RADII
            ]

            print(
                f"K={k:3d} | "
                + " | ".join(
                    f"{radius}m: {value}"
                    for radius, value in zip(RADII, values)
                )
            )


if __name__ == "__main__":
    main()

