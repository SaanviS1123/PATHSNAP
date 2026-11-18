
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

df = pd.read_csv(
    "all_training_candidates.csv",
    dtype={"trip_id": str}
)

features = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

X = df[features].fillna(0)
y = df["is_correct_arc"]
groups = df["trip_id"]

# Load parameters only; models will be refitted
# within each fold to prevent validation leakage.
tuned = joblib.load("pathsnap_tuned_model.joblib")

configs = {
    "Untuned": {
        "max_iter": 100,
        "max_depth": 6,
        "random_state": 42
    },
    "Tuned": tuned.get_params()
}

results = []
cv = GroupKFold(n_splits=5)

for name, params in configs.items():
    print(f"\n{name} model", flush=True)

    for fold, (tr, va) in enumerate(
        cv.split(X, y, groups), 1
    ):
        model = HistGradientBoostingClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr])

        probabilities = model.predict_proba(
            X.iloc[va]
        )[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        val = df.iloc[va][
            ["trip_id", "observation_id", "is_correct_arc"]
        ].copy()

        val["score"] = probabilities

        winners = val.loc[
            val.groupby(
                ["trip_id", "observation_id"]
            )["score"].idxmax()
        ]

        top1 = winners["is_correct_arc"].mean()

        results.append({
            "Model": name,
            "Fold": fold,
            "Top1": top1,
            "F1": f1_score(
                y.iloc[va],
                predictions,
                zero_division=0
            ),
            "LogLoss": log_loss(
                y.iloc[va],
                probabilities,
                labels=[0, 1]
            )
        })

        print(
            f"  Fold {fold}: "
            f"Top1={top1:.4f} "
            f"F1={results[-1]['F1']:.4f}",
            flush=True
        )

results_df = pd.DataFrame(results)
results_df.to_csv(
    "tuned_vs_untuned_folds.csv",
    index=False
)

summary = results_df.groupby("Model")[
    ["Top1", "F1", "LogLoss"]
].mean()

print("\nTUNED VS UNTUNED COMPARISON")
print(summary.round(4).to_string())

summary.to_csv("tuned_vs_untuned_summary.csv")
