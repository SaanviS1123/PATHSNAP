
from pathlib import Path
import json

import pandas as pd

from src.data_processing.loaders import load_trip
from src.features.build import (
    build_trip_features,
    FEATURE_COLUMNS
)

ROOT = Path("Student folder") / "train"
OUTPUT = Path("tests/fixtures")
OUTPUT.mkdir(parents=True, exist_ok=True)

TRIP_COUNT = 5
MAX_OBSERVATIONS = 100


def main():
    labels = pd.read_parquet(
        "data/processed/obs_labels.parquet"
    )

    labels["trip_id"] = (
        labels["trip_id"].astype(str).str.zfill(8)
    )

    folders = sorted(
        folder for folder in ROOT.iterdir()
        if folder.is_dir()
    )[:TRIP_COUNT]

    frames = []
    metadata = []

    print("A6 FIVE-TRIP FIXTURE")

    for folder in folders:
        trip = load_trip(folder)

        # Generate features with the same pipeline
        # that will be used for the full dataset.
        trip_labels = labels[
            labels["trip_id"] == trip["trip_id"]
        ]

        features = build_trip_features(
            trip,
            labels=trip_labels
        )

        # Retain the first 100 observations
        # to keep the fixture lightweight.
        features = features[
            features["observation_id"]
            < MAX_OBSERVATIONS
        ].copy()

        frames.append(features)

        metadata.append({
            "trip_id": trip["trip_id"],
            "observations": int(
                features["observation_id"].nunique()
            ),
            "candidate_rows": len(features),
            "positive_labels": int(
                features["is_correct_arc"].sum()
            )
        })

        print(
            f"Trip {trip['trip_id']}: "
            f"{len(features)} candidate rows"
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    expected_columns = [
        "trip_id",
        "observation_id",
        "arc_id",
        *FEATURE_COLUMNS,
        "is_correct_arc"
    ]

    assert combined.columns.tolist() == expected_columns
    assert combined[FEATURE_COLUMNS].notna().all().all()
    assert combined["trip_id"].nunique() == TRIP_COUNT

    combined.to_parquet(
        OUTPUT / "five_trip_features.parquet",
        index=False
    )

    with open(
        OUTPUT / "fixture_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "description": (
                    "First 100 observations from each "
                    "of five training trips"
                ),
                "trips": metadata,
                "feature_columns": FEATURE_COLUMNS
            },
            file,
            indent=2
        )

    print("\nFIXTURE SUMMARY")
    print("Trips:", combined["trip_id"].nunique())
    print(
        "Observations:",
        combined.groupby(
            ["trip_id", "observation_id"]
        ).ngroups
    )
    print("Candidate rows:", len(combined))
    print(
        "Positive labels:",
        int(combined["is_correct_arc"].sum())
    )

    print("\nSaved:")
    print(OUTPUT / "five_trip_features.parquet")
    print(OUTPUT / "fixture_metadata.json")
    print("\nA6 FIXTURE VALIDATION PASSED")


if __name__ == "__main__":
    main()
