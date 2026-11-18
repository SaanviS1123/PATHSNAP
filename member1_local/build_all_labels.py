
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data_processing.loaders import load_trip
from src.features.labels import align_route_to_gps

TRAIN_ROOT = Path("Student folder") / "train"
DATA_DIR = Path("data/processed")
METRICS_DIR = Path("results/metrics")

DATA_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    started = time.perf_counter()

    all_labels = []
    summaries = []

    folders = sorted(
        folder for folder in TRAIN_ROOT.iterdir()
        if folder.is_dir()
    )

    print("A4 — FULL LABEL ALIGNMENT")
    print("Training trips:", len(folders))

    for index, folder in enumerate(folders, 1):
        trip = load_trip(folder)

        labels = align_route_to_gps(
            trip["track"],
            trip["nodes"],
            trip["arcs"],
            trip["route"]
        )

        labels.insert(
            0, "trip_id", trip["trip_id"]
        )

        route = trip["route"]
        arcs = trip["arcs"]

        disconnected = int(np.sum(
            arcs[route[:-1], 1]
            != arcs[route[1:], 0]
        ))

        summaries.append({
            "trip_id": trip["trip_id"],
            "gps_points": len(labels),
            "route_arcs": len(route),
            "unique_assigned_arcs": (
                labels["arc_id"].nunique()
            ),
            "median_distance_m": (
                labels["distance_m"].median()
            ),
            "p95_distance_m": (
                labels["distance_m"].quantile(0.95)
            ),
            "max_distance_m": (
                labels["distance_m"].max()
            ),
            "points_over_30m": int(
                (labels["distance_m"] > 30).sum()
            ),
            "points_over_50m": int(
                (labels["distance_m"] > 50).sum()
            ),
            "backward_jumps": int(
                labels["backward_jump"].sum()
            ),
            "disconnected_reference_links": (
                disconnected
            )
        })

        all_labels.append(labels)

        if index % 10 == 0 or index == len(folders):
            print(
                f"Completed {index}/{len(folders)} trips"
            )

    combined = pd.concat(
        all_labels,
        ignore_index=True
    )

    summary = pd.DataFrame(summaries)

    labels_path = DATA_DIR / "obs_labels.parquet"
    summary_path = (
        METRICS_DIR / "label_alignment.csv"
    )

    combined.to_parquet(
        labels_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print("\nRESULTS")
    print("Total labeled GPS points:", len(combined))
    print(
        "Overall median distance (m):",
        round(combined["distance_m"].median(), 2)
    )
    print(
        "Points over 50 m:",
        int((combined["distance_m"] > 50).sum())
    )
    print(
        "Total backward jumps:",
        int(combined["backward_jump"].sum())
    )
    print(
        "Disconnected reference links:",
        int(
            summary[
                "disconnected_reference_links"
            ].sum()
        )
    )
    print(
        "Elapsed seconds:",
        round(time.perf_counter() - started, 2)
    )

    print("\nSaved:")
    print(labels_path)
    print(summary_path)


if __name__ == "__main__":
    main()
