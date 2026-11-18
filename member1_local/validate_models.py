
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)

df = pd.read_csv("all_training_candidates.csv")

features = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

# Preserve all observations, including those with no positive candidate.
# Missing positives are counted as Top-1 errors.
X = df[features].fillna(0)
y = df["is_correct_arc"]
groups = df["trip_id"]

models = {
    "Logistic Regression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42
        )
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),
    "Extra Trees": lambda: ExtraTreesClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ),
    "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        random_state=42
    )
}

cv = GroupKFold(n_splits=5)
results = []

print("STARTING GROUP-AWARE 5-FOLD VALIDATION", flush=True)
print("Rows:", len(df), "| Trips:", groups.nunique(), flush=True)

for name, create_model in models.items():
    print(f"\nTraining {name}...", flush=True)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        model = create_model()

        start = time.time()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_seconds = time.time() - start

        start = time.time()
        probabilities = model.predict_proba(X.iloc[val_idx])[:, 1]
        inference_seconds = time.time() - start

        predictions = (probabilities >= 0.5).astype(int)

        validation = df.iloc[val_idx][
            ["trip_id", "observation_id", "is_correct_arc"]
        ].copy()

        validation["probability"] = probabilities

        # Highest-scoring candidate for each GPS observation.
        best_rows = validation.loc[
            validation.groupby(
                ["trip_id", "observation_id"]
            )["probability"].idxmax()
        ]

        top1 = best_rows["is_correct_arc"].mean()

        # Fraction of observations whose provisional label
        # is actually present among the generated candidates.
        coverage = validation.groupby(
            ["trip_id", "observation_id"]
        )["is_correct_arc"].max().mean()

        results.append({
            "Model": name,
            "Fold": fold,
            "Accuracy": accuracy_score(
                y.iloc[val_idx], predictions
            ),
            "Precision": precision_score(
                y.iloc[val_idx], predictions, zero_division=0
            ),
            "Recall": recall_score(
                y.iloc[val_idx], predictions, zero_division=0
            ),
            "F1": f1_score(
                y.iloc[val_idx], predictions, zero_division=0
            ),
            "Log Loss": log_loss(
                y.iloc[val_idx],
                probabilities,
                labels=[0, 1]
            ),
            "Top-1 Accuracy": top1,
            "Candidate Coverage": coverage,
            "Training Seconds": train_seconds,
            "Inference Seconds": inference_seconds
        })

        print(
            f"  Fold {fold}: "
            f"F1={results[-1]['F1']:.4f}, "
            f"Top1={top1:.4f}, "
            f"Coverage={coverage:.4f}",
            flush=True
        )

results_df = pd.DataFrame(results)
results_df.to_csv("group_validation_folds.csv", index=False)

summary = results_df.groupby("Model").agg({
    "Accuracy": "mean",
    "Precision": "mean",
    "Recall": "mean",
    "F1": "mean",
    "Log Loss": "mean",
    "Top-1 Accuracy": "mean",
    "Candidate Coverage": "mean",
    "Training Seconds": "mean",
    "Inference Seconds": "mean"
}).sort_values("Top-1 Accuracy", ascending=False)

summary.to_csv("group_validation_summary.csv")

print("\nFINAL GROUP-AWARE MODEL COMPARISON")
print(summary.round(4).to_string())
print("\nSaved group_validation_folds.csv")
print("Saved group_validation_summary.csv")
