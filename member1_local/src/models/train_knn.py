
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(".")
DATA_PATH = ROOT / "data/processed/train.parquet"
SCHEMA_PATH = ROOT / "models/feature_schema.json"
FOLDS_PATH = ROOT / "configs/folds.json"

MODEL_PATH = ROOT / "models/knn_model.joblib"
METRICS_PATH = ROOT / "results/metrics/knn_metrics.json"

SEED = 42
FOLD_NUMBER = 1

# Limit training size to keep KNN manageable.
MAX_TRAIN_POSITIVES = 30000
MAX_TRAIN_NEGATIVES = 90000

# Validation is sampled by observation, not candidate row.
MAX_VALIDATION_OBSERVATIONS = 10000

def sample_training_rows(frame):
    positives = frame[
        frame["is_correct_arc"] == 1
    ]
    negatives = frame[
        frame["is_correct_arc"] == 0
    ]

    positives = positives.sample(
        n=min(len(positives), MAX_TRAIN_POSITIVES),
        random_state=SEED
    )

    negatives = negatives.sample(
        n=min(len(negatives), MAX_TRAIN_NEGATIVES),
        random_state=SEED
    )

    return pd.concat(
        [positives, negatives],
        ignore_index=True
    ).sample(frac=1, random_state=SEED)

def sample_validation_observations(frame):
    # Keep every candidate for each sampled GPS observation.
    observations = frame[
        ["trip_id", "observation_id"]
    ].drop_duplicates()

    if len(observations) > MAX_VALIDATION_OBSERVATIONS:
        observations = observations.sample(
            n=MAX_VALIDATION_OBSERVATIONS,
            random_state=SEED
        )

    return frame.merge(
        observations,
        on=["trip_id", "observation_id"],
        how="inner"
    )

def main():
    started = time.perf_counter()

    schema = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )
    folds = json.loads(
        FOLDS_PATH.read_text(encoding="utf-8")
    )

    feature_columns = schema["feature_columns"]

    fold = next(
        f for f in folds["folds"]
        if f["fold"] == FOLD_NUMBER
    )

    print("A9 KNN BASELINE")
    print("Validation fold:", FOLD_NUMBER)
    print("Features:", feature_columns)

    columns = [
        "trip_id",
        "observation_id",
        *feature_columns,
        "is_correct_arc"
    ]

    data = pd.read_parquet(
        DATA_PATH,
        columns=columns
    )

    data["trip_id"] = (
        data["trip_id"].astype(str).str.zfill(8)
    )

    train_ids = set(fold["train_trip_ids"])
    val_ids = set(fold["validation_trip_ids"])

    assert train_ids.isdisjoint(val_ids)

    training = data[
        data["trip_id"].isin(train_ids)
    ]
    validation = data[
        data["trip_id"].isin(val_ids)
    ]

    if training.empty or validation.empty:
        raise ValueError(
            "Empty train/validation split. Check folds.json."
        )

    train_sample = sample_training_rows(training)
    val_sample = sample_validation_observations(validation)

    print("Training candidate rows:", len(train_sample))
    print("Validation candidate rows:", len(val_sample))
    print(
        "Validation GPS observations:",
        val_sample.groupby(
            ["trip_id", "observation_id"]
        ).ngroups
    )

    X_train = train_sample[feature_columns]
    y_train = train_sample["is_correct_arc"].astype(int)

    X_val = val_sample[feature_columns]
    y_val = val_sample["is_correct_arc"].astype(int)

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                algorithm="auto",
                n_jobs=-1
            )
        )
    ])

    print("\nTraining KNN...")
    model.fit(X_train, y_train)

    print("Predicting validation probabilities...")
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # GPS-level top-1:
    # select the candidate with the highest probability
    # for each observation.
    scored = val_sample[
        ["trip_id", "observation_id", "is_correct_arc"]
    ].copy()

    scored["probability"] = probabilities

    best_indices = scored.groupby(
        ["trip_id", "observation_id"]
    )["probability"].idxmax()

    selected = scored.loc[best_indices]

    top1_accuracy = float(
        selected["is_correct_arc"].mean()
    )

    metrics = {
        "model": "KNN",
        "fold": FOLD_NUMBER,
        "seed": SEED,
        "training_candidate_rows": len(train_sample),
        "validation_candidate_rows": len(val_sample),
        "validation_observations": len(selected),
        "precision": float(
            precision_score(
                y_val, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_val, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_val, predictions, zero_division=0
            )
        ),
        "candidate_accuracy": float(
            accuracy_score(y_val, predictions)
        ),
        "log_loss": float(
            log_loss(
                y_val,
                np.clip(probabilities, 1e-7, 1 - 1e-7),
                labels=[0, 1]
            )
        ),
        "gps_top1_accuracy": top1_accuracy,
        "validation_positive_labels": int(y_val.sum()),
        "elapsed_seconds": round(
            time.perf_counter() - started, 2
        )
    }

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    METRICS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "schema_version": schema["schema_version"]
        },
        MODEL_PATH
    )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nKNN VALIDATION METRICS")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("\nA9 KNN TRAINING COMPLETE")

if __name__ == "__main__":
    main()