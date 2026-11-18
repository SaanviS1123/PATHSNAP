
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(".")
TRAIN = ROOT / "data/processed/train.parquet"
TEST = ROOT / "data/processed/student_test.parquet"
SCHEMA = ROOT / "models/feature_schema.json"


def check_dataset(path, features, is_train):
    table = pq.read_table(path)
    df = table.to_pandas()

    print(f"\nChecking {path}")
    print("Candidate rows:", len(df))
    print("Trips:", df["trip_id"].nunique())

    required = [
        "trip_id",
        "observation_id",
        "arc_id",
        *features
    ]

    if is_train:
        required.append("is_correct_arc")

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    assert not missing_columns, (
        f"Missing columns: {missing_columns}"
    )

    if not is_train:
        assert "is_correct_arc" not in df.columns, (
            "Test data contains training labels!"
        )

    null_counts = df[required].isna().sum()
    assert null_counts.sum() == 0, (
        f"Missing values:\n{null_counts[null_counts > 0]}"
    )

    duplicates = df.duplicated(
        subset=["trip_id", "observation_id", "arc_id"]
    ).sum()

    assert duplicates == 0, (
        f"Duplicate candidate rows: {duplicates}"
    )

    if is_train:
        assert df["is_correct_arc"].isin([0, 1]).all()

        positives_per_observation = df.groupby(
            ["trip_id", "observation_id"]
        )["is_correct_arc"].sum()

        assert (
            positives_per_observation <= 1
        ).all(), "Multiple positive candidates for one observation!"

        print(
            "Observations with positive candidate:",
            int((positives_per_observation == 1).sum())
        )
        print(
            "Observations without positive candidate:",
            int((positives_per_observation == 0).sum())
        )

    observations = df.groupby(
        ["trip_id", "observation_id"]
    ).size()

    assert (observations >= 1).all(), (
        "An observation has no candidates!"
    )

    print("GPS observations:", len(observations))
    print("Duplicate candidates:", duplicates)
    print("Missing feature values: 0")
    print("Schema: PASS")
    print("Validation: PASS")


def main():
    schema = json.loads(
        SCHEMA.read_text(encoding="utf-8")
    )

    features = schema["feature_columns"]

    print("A11 FEATURE VALIDATION")
    print("Features:", features)

    check_dataset(TRAIN, features, is_train=True)
    check_dataset(TEST, features, is_train=False)

    print("\nALL FEATURE CHECKS PASSED")


if __name__ == "__main__":
    main()
