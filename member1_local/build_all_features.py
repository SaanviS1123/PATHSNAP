
from pathlib import Path
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_processing.loaders import load_trip
from src.features.build import build_trip_features, FEATURE_COLUMNS

DATA_ROOT = Path("Student folder")
OUTPUT_ROOT = Path("data/processed")
REPORT_PATH = Path("results/metrics/feature_build_report.csv")

ID_COLUMNS = ["trip_id", "observation_id", "arc_id"]
TRAIN_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + ["is_correct_arc"]
TEST_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS


def write_dataset(split, labels=None):
    root = DATA_ROOT / split
    output = OUTPUT_ROOT / f"{split}.parquet"

    folders = sorted(
        p for p in root.iterdir() if p.is_dir()
    )

    writer = None
    report_rows = []
    total_rows = 0
    started = time.perf_counter()

    try:
        for number, folder in enumerate(folders, start=1):
            trip = load_trip(folder)
            trip_id = trip["trip_id"]

            trip_labels = None
            if labels is not None:
                trip_labels = labels[
                    labels["trip_id"] == trip_id
                ]

                if len(trip_labels) != len(trip["track"]):
                    raise ValueError(
                        f"Label count mismatch: {trip_id}"
                    )

            features = build_trip_features(
                trip,
                labels=trip_labels,
                search_k=500,
                candidate_k=30,
                radius_m=100
            )

            expected_observations = len(trip["track"])
            actual_observations = (
                features["observation_id"].nunique()
            )

            if actual_observations != expected_observations:
                raise ValueError(
                    f"Missing observations in {trip_id}: "
                    f"{actual_observations}/"
                    f"{expected_observations}"
                )

            columns = (
                TRAIN_COLUMNS if labels is not None
                else TEST_COLUMNS
            )

            features = features[columns]

            if features[FEATURE_COLUMNS].isna().any().any():
                raise ValueError(
                    f"Missing feature values in {trip_id}"
                )

            table = pa.Table.from_pandas(
                features,
                preserve_index=False
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output),
                    table.schema,
                    compression="snappy"
                )

            writer.write_table(table)

            positive_labels = (
                int(features["is_correct_arc"].sum())
                if labels is not None else None
            )

            report_rows.append({
                "split": split,
                "trip_id": trip_id,
                "gps_observations": expected_observations,
                "candidate_rows": len(features),
                "positive_labels": positive_labels,
                "candidate_coverage": (
                    positive_labels / expected_observations
                    if positive_labels is not None
                    else None
                )
            })

            total_rows += len(features)

            if number % 10 == 0 or number == len(folders):
                print(
                    f"{split}: {number}/{len(folders)} trips, "
                    f"{total_rows:,} candidate rows"
                )

    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"No trips found in {root}")

    print(
        f"Saved {output} in "
        f"{time.perf_counter() - started:.1f}s"
    )

    return report_rows


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    labels = pd.read_parquet(
        OUTPUT_ROOT / "obs_labels.parquet"
    )

    labels["trip_id"] = (
        labels["trip_id"].astype(str).str.zfill(8)
    )

    print("A7 FULL FEATURE GENERATION")
    print("Feature columns:", FEATURE_COLUMNS)

    report = []
    report.extend(
        write_dataset("train", labels=labels)
    )
    report.extend(
        write_dataset("student_test", labels=None)
    )

    summary = pd.DataFrame(report)
    summary.to_csv(REPORT_PATH, index=False)

    print("\nFINAL SUMMARY")

    for split, group in summary.groupby("split"):
        print(
            f"{split}: "
            f"{len(group)} trips, "
            f"{group['gps_observations'].sum():,} GPS, "
            f"{group['candidate_rows'].sum():,} "
            f"candidate rows"
        )

        if split == "train":
            print(
                "Provisional label coverage:",
                f"{group['positive_labels'].sum() / group['gps_observations'].sum():.2%}"
            )

    print("\nSaved:", REPORT_PATH)
    print("A7 FEATURE GENERATION COMPLETE")


if __name__ == "__main__":
    main()
