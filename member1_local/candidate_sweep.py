
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data_processing.loaders import load_trip
from src.data_processing.geo import LocalMetricProjection
from src.features.candidates import RoadCandidateIndex

TRAIN_ROOT = Path("Student folder") / "train"
OUTPUT = Path("results/metrics/candidate_sweep.csv")

K_VALUES = [5, 10, 20, 30]
SEARCH_K_VALUES = [100, 300, 500]
RADIUS_VALUES = [10, 20, 30, 50, 100]

SAMPLE_PER_TRIP = 200


def main():
    labels = pd.read_parquet(
        "data/processed/obs_labels.parquet"
    )

    labels["trip_id"] = (
        labels["trip_id"].astype(str).str.zfill(8)
    )

    results = []

    folders = sorted(
        folder for folder in TRAIN_ROOT.iterdir()
        if folder.is_dir()
    )

    print("A5 CANDIDATE SWEEP")
    print("Training trips:", len(folders))

    started = time.perf_counter()

    for trip_number, folder in enumerate(folders, 1):
        trip = load_trip(folder)

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
            labels[
                labels["trip_id"] == folder.name
            ]
            .set_index("observation_id")
        )

        sample_ids = np.linspace(
            0,
            len(gps_xy) - 1,
            min(SAMPLE_PER_TRIP, len(gps_xy)),
            dtype=int
        )

        for obs_id in sample_ids:
            point = gps_xy[obs_id]
            true_arc = int(
                trip_labels.loc[
                    obs_id, "arc_id"
                ]
            )

            for search_k in SEARCH_K_VALUES:
                ids, distances = index.query(
                    point,
                    k=max(K_VALUES),
                    search_k=search_k
                )

                for k in K_VALUES:
                    selected_ids = ids[:k]
                    selected_distances = distances[:k]

                    for radius in RADIUS_VALUES:
                        valid = (
                            selected_distances <= radius
                        )

                        covered = bool(
                            true_arc in selected_ids[valid]
                        )

                        results.append({
                            "trip_id": folder.name,
                            "observation_id": int(obs_id),
                            "search_k": search_k,
                            "k": k,
                            "radius_m": radius,
                            "covered": covered
                        })

        if trip_number % 10 == 0:
            print(
                f"Completed {trip_number}/{len(folders)} trips"
            )

    df = pd.DataFrame(results)

    summary = (
        df.groupby(
            ["search_k", "k", "radius_m"],
            as_index=False
        )
        .agg(
            observations=("covered", "size"),
            coverage=("covered", "mean")
        )
    )

    summary["miss_rate"] = (
        1 - summary["coverage"]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        OUTPUT,
        index=False
    )

    print("\nTOP CONFIGURATIONS")
    print(
        summary.sort_values(
            "miss_rate"
        ).head(10).to_string(index=False)
    )

    print(
        "\nElapsed seconds:",
        round(time.perf_counter() - started, 2)
    )

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
