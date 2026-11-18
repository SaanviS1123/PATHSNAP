
import time
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("candidate_training_trip1.csv")

features = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

valid = df.groupby("observation_id")[
    "is_correct_arc"
].transform("max") == 1

df = df[valid].copy()

X = df[features]
y = df["is_correct_arc"]

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=100,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        random_state=42
    )
}

results = []

for name, model in models.items():
    start = time.time()
    model.fit(X, y)
    training_time = time.time() - start

    pred = model.predict(X)

    results.append({
        "Model": name,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Training Time (s)": round(training_time, 2)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("F1", ascending=False)

print("\nFIVE MODEL TRAINING COMPARISON")
print("(Training-set metrics only — not validation)")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison_training.csv", index=False)
print("\nSaved model_comparison_training.csv")
