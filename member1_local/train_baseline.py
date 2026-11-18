
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("candidate_training_trip1.csv")

feature_cols = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

# Exclude observations without a positive candidate
valid_obs = df.groupby("observation_id")["is_correct_arc"].transform("max") == 1
df = df[valid_obs].copy()

X = df[feature_cols]
y = df["is_correct_arc"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X, y)

predictions = model.predict(X)

print("BASELINE PIPELINE SANITY CHECK")
print("Training rows:", len(df))
print("Positive rows:", int(y.sum()))
print("\nTRAINING-SET METRICS (NOT VALIDATION)")
print(classification_report(y, predictions, zero_division=0))

joblib.dump(model, "baseline_model.joblib")
print("Saved baseline_model.joblib")

