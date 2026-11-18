
import json
import time
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("all_training_candidates.csv")

features = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

X = df[features].fillna(0)
y = df["is_correct_arc"]

model = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=6,
    random_state=42
)

start = time.time()
model.fit(X, y)
elapsed = time.time() - start

joblib.dump(model, "pathsnap_candidate_model.joblib")

metadata = {
    "model": "HistGradientBoostingClassifier",
    "features": features,
    "training_trips": int(df["trip_id"].nunique()),
    "training_rows": int(len(df)),
    "positive_labels": int(y.sum()),
    "training_seconds": round(elapsed, 2),
    "label_type": "route-derived pseudo-labels",
    "status": "untuned baseline"
}

with open("pathsnap_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("SELECTED MODEL TRAINED")
print("Model: HistGradientBoosting")
print("Training trips:", metadata["training_trips"])
print("Training rows:", metadata["training_rows"])
print("Training time:", metadata["training_seconds"], "seconds")
print("Saved pathsnap_candidate_model.joblib")
print("Saved pathsnap_model_metadata.json")
