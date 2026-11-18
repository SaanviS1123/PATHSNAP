"""Group-aware Optuna TPE search for the Random Forest classifier."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.harness import evaluate_model
from src.models.training import load_training_data

ROOT = Path(__file__).resolve().parents[3]


def tune(data_path=None, trials: int = 30, seed: int = 42, output_path=None):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optuna to run Bayesian tuning (pip install optuna).") from exc
    frame, features = load_training_data(data_path) if data_path else load_training_data()
    started = time.perf_counter()
    trial_start = started
    labels_path = ROOT / "data" / "processed" / "obs_labels.parquet"
    obs_labels = pd.read_parquet(labels_path) if labels_path.exists() else None
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed), study_name="pathsnap-rf-top1")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 20, 32]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
            "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": seed,
        }
        model = RandomForestClassifier(**params)
        scores = evaluate_model(model, "random_forest_tpe_trial", frame,
            feature_columns=features, config=params, random_seed=seed,
            obs_labels=obs_labels, log=False)
        trial.set_user_attr("fold_metrics_json", scores["fold_metrics_json"])
        trial.set_user_attr("compute_seconds", time.perf_counter() - trial_start)
        return scores["top1_accuracy"]

    def timed_objective(trial):
        nonlocal trial_start
        trial_start = time.perf_counter()
        return objective(trial)

    study.optimize(timed_objective, n_trials=trials)
    elapsed = time.perf_counter() - started
    rows = []
    for trial in study.trials:
        rows.append({"trial": trial.number, "state": trial.state.name,
            "objective_top1_accuracy": trial.value, "params": trial.params,
            "fold_metrics_json": trial.user_attrs.get("fold_metrics_json"),
            "compute_seconds": trial.user_attrs.get("compute_seconds"),
            "best_trial": trial.number == study.best_trial.number})
    path = Path(output_path) if output_path else ROOT / "results" / "metrics" / "tuning_trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    summary = {"objective": "trip-fold mean observation Top-1 accuracy",
        "best_trial": study.best_trial.number, "best_top1_accuracy": study.best_value,
        "best_params": study.best_params, "n_trials": len(study.trials), "total_compute_seconds": elapsed}
    import json
    (path.parent / "tuning_best.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(tune(args.data, args.trials, args.seed))
