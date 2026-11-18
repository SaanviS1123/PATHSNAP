
import time
import json
import joblib
import optuna
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = [
    "distance_m",
    "segment_length_m",
    "bearing_difference",
    "speed_mps",
    "time_delta_s"
]

df = pd.read_csv(
    "all_training_candidates.csv",
    dtype={"trip_id": str}
)

X = df[FEATURES].fillna(0).reset_index(drop=True)
y = df["is_correct_arc"].astype(int).reset_index(drop=True)
groups = df["trip_id"].reset_index(drop=True)

# Unique observation identifier within each trip.
obs_keys = (
    groups + "_" +
    df["observation_id"].astype(str)
).reset_index(drop=True)

cv = GroupKFold(n_splits=3)
folds = list(cv.split(X, y, groups))

print("BAYESIAN OPTIMIZATION")
print("Training rows:", len(X))
print("Training trips:", groups.nunique())
print("Validation folds:", len(folds))
print("Optimization trials: 15", flush=True)

def top1_accuracy(model, val_idx):
    probabilities = model.predict_proba(
        X.iloc[val_idx]
    )[:, 1]

    validation = pd.DataFrame({
        "observation": obs_keys.iloc[val_idx].to_numpy(),
        "correct": y.iloc[val_idx].to_numpy(),
        "score": probabilities
    })

    winners = validation.loc[
        validation.groupby("observation")["score"].idxmax()
    ]

    return float(winners["correct"].mean())

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.2, log=True
        ),
        "max_iter": trial.suggest_int(
            "max_iter", 60, 180
        ),
        "max_leaf_nodes": trial.suggest_int(
            "max_leaf_nodes", 15, 63
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 12
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 10, 80
        ),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-4, 10.0, log=True
        ),
        "random_state": 42
    }

    scores = []

    for train_idx, val_idx in folds:
        model = HistGradientBoostingClassifier(**params)

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx]
        )

        score = top1_accuracy(model, val_idx)
        scores.append(score)

    mean_score = float(np.mean(scores))

    print(
        f"Trial {trial.number + 1:02d} | "
        f"Top-1={mean_score:.4f}",
        flush=True
    )

    return mean_score

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

start = time.time()

study.optimize(
    objective,
    n_trials=15,
    show_progress_bar=False
)

elapsed = time.time() - start

print("\nOPTIMIZATION COMPLETE")
print("Best Top-1 accuracy:", round(study.best_value, 4))
print("Best parameters:")

for key, value in study.best_params.items():
    print(f"  {key}: {value}")

print("Optimization time:", round(elapsed, 1), "seconds")

# Save every trial for the evaluation report.
trials = study.trials_dataframe()
trials.to_csv("bayesian_trials.csv", index=False)

# Fit the tuned model on all permitted training trips.
best_model = HistGradientBoostingClassifier(
    **study.best_params,
    random_state=42
)

best_model.fit(X, y)

joblib.dump(
    best_model,
    "pathsnap_tuned_model.joblib"
)

with open("pathsnap_tuned_metadata.json", "w") as f:
    json.dump({
        "model": "HistGradientBoostingClassifier",
        "features": FEATURES,
        "best_parameters": study.best_params,
        "best_cv_top1": study.best_value,
        "cv_folds": 3,
        "optimization_trials": 15,
        "optimization_seconds": elapsed,
        "label_type": "route-derived pseudo-labels"
    }, f, indent=2)

print("\nSaved:")
print("  bayesian_trials.csv")
print("  pathsnap_tuned_model.joblib")
print("  pathsnap_tuned_metadata.json")
