
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.loaders import load_trip
from src.data_processing.geo import LocalMetricProjection
from src.features.candidates import RoadCandidateIndex

TRIP_IDS = ["00000001", "00000033", "00000054"]
SAMPLE_PER_TRIP = 1000


def main():
    print("CANDIDATE DENSITY DIAGNOSIS")

    for trip_id in TRIP_IDS:
        trip = load_trip(
            Path("Student folder") / "train" / trip_id
        )

        projection = LocalMetricProjection(trip["nodes"])
        gps_xy = projection.transform(trip["track"][:, :2])

        index = RoadCandidateIndex(
            trip["nodes"],
            trip["arcs"],
            projection
        )

        sample_ids = np.linspace(
            0,
            len(gps_xy) - 1,
            min(SAMPLE_PER_TRIP, len(gps_xy)),
            dtype=int
        )

        counts_100m = []
        counts_200m = []
        nearest_distances = []

        for obs_id in sample_ids:
            _, distances = index.query(
                gps_xy[obs_id],
                k=100,
                search_k=2000
            )

            counts_100m.append(
                int(np.sum(distances <= 100))
            )

            counts_200m.append(
                int(np.sum(distances <= 200))
            )

            nearest_distances.append(
                float(distances[0])
            )

        counts_100m = np.array(counts_100m)
        counts_200m = np.array(counts_200m)

        print(f"\nTrip {trip_id}")
        print(
            "Median candidates within 100 m:",
            np.median(counts_100m)
        )
        print(
            "Median candidates within 200 m:",
            np.median(counts_200m)
        )
        print(
            "Observations with fewer than 10 "
            "candidates within 100 m:",
            f"{np.mean(counts_100m < 10):.2%}"
        )
        print(
            "Observations with 30 or more "
            "candidates within 100 m:",
            f"{np.mean(counts_100m >= 30):.2%}"
        )
        print(
            "Median nearest-road distance:",
            f"{np.median(nearest_distances):.1f} m"
        )


if __name__ == "__main__":
    main()
