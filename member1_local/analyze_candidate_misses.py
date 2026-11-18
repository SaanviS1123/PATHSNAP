
from pathlib import Path
import pandas as pd

INPUT = Path("results/metrics/feature_build_report.csv")
OUTPUT = Path("results/metrics/candidate_miss_analysis.csv")


def main():
    report = pd.read_csv(INPUT)

    train = report[
        report["split"] == "train"
    ].copy()

    train["trip_id"] = (
        train["trip_id"]
        .astype(str)
        .str.zfill(8)
    )

    train["missing_positive_observations"] = (
        train["gps_observations"]
        - train["positive_labels"]
    ).astype(int)

    train["candidate_miss_rate"] = (
        train["missing_positive_observations"]
        / train["gps_observations"]
    )

    train = train.sort_values(
        "missing_positive_observations",
        ascending=False
    )

    total_gps = int(
        train["gps_observations"].sum()
    )

    total_positive = int(
        train["positive_labels"].sum()
    )

    total_missing = int(
        train["missing_positive_observations"].sum()
    )

    train["share_of_total_misses"] = (
        train["missing_positive_observations"]
        / total_missing
        if total_missing else 0.0
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        OUTPUT,
        index=False
    )

    print("A11 CANDIDATE-MISS ANALYSIS")
    print("Training trips:", len(train))
    print("Total GPS observations:", total_gps)
    print("Positive candidate observations:", total_positive)
    print("Missing positive observations:", total_missing)

    print(
        "Overall provisional label coverage:",
        f"{total_positive / total_gps:.2%}"
    )

    print(
        "Overall candidate miss rate:",
        f"{total_missing / total_gps:.2%}"
    )

    print("\nTOP 10 TRIPS BY MISSING POSITIVES")

    print(
        train[
            [
                "trip_id",
                "gps_observations",
                "missing_positive_observations",
                "candidate_miss_rate",
                "share_of_total_misses"
            ]
        ].head(10).to_string(index=False)
    )

    difficult = train[
        train["trip_id"].isin(
            ["00000033", "00000054"]
        )
    ]

    print(
        "\nShare of misses from Trips 33 and 54:",
        f"{difficult['missing_positive_observations'].sum() / total_missing:.2%}"
    )

    print("\nSaved:", OUTPUT)
    print("A11 ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
