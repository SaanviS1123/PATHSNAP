
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPORT = Path("results/metrics/feature_build_report.csv")
TRAIN_DATA = Path("data/processed/train.parquet")
OUTPUT = Path("results/plots")

OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    report = pd.read_csv(REPORT)
    train = report[
        report["split"] == "train"
    ].copy()

    train["trip_id"] = (
        train["trip_id"].astype(str).str.zfill(8)
    )

    print("A8 EDA PLOT GENERATION")

    # Plot 1: GPS observations per trip
    ordered = train.sort_values(
        "gps_observations",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(
        range(len(ordered)),
        ordered["gps_observations"]
    )
    ax.set_title("GPS Observations per Training Trip")
    ax.set_xlabel("Training trips, sorted by size")
    ax.set_ylabel("GPS observations")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "eda_trip_sizes.png",
        dpi=160
    )
    plt.close(fig)

    # Plot 2: Candidate-label coverage per trip
    ordered = train.sort_values(
        "candidate_coverage"
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(
        range(len(ordered)),
        ordered["candidate_coverage"] * 100
    )
    ax.axhline(
        90,
        linestyle="--",
        label="90% coverage"
    )
    ax.set_title(
        "Provisional Candidate-Label Coverage by Trip"
    )
    ax.set_xlabel("Training trips, sorted by coverage")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "eda_candidate_coverage.png",
        dpi=160
    )
    plt.close(fig)

    # Plot 3: Candidate distance distribution.
    # Read only the distance column to reduce memory.
    distances = pd.read_parquet(
        TRAIN_DATA,
        columns=["distance_m"]
    )["distance_m"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        distances.clip(upper=200),
        bins=60
    )
    ax.set_title(
        "GPS-to-Candidate Road Distance Distribution"
    )
    ax.set_xlabel(
        "Distance (m), clipped at 200 m"
    )
    ax.set_ylabel("Candidate rows")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "eda_candidate_distances.png",
        dpi=160
    )
    plt.close(fig)

    print("Training trips:", len(train))
    print(
        "Median GPS observations per trip:",
        int(train["gps_observations"].median())
    )
    print(
        "Median trip-level candidate coverage:",
        f"{train['candidate_coverage'].median():.2%}"
    )
    print(
        "Median candidate distance:",
        f"{distances.median():.2f} m"
    )

    print("\nSaved plots:")
    for path in sorted(OUTPUT.glob("eda_*.png")):
        print(path)

    print("\nA8 PLOTS COMPLETE")


if __name__ == "__main__":
    main()
