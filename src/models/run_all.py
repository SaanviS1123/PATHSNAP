"""Run Member 2 model CV, comparison, and final artifact export."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.harness import evaluate_model
from src.models.training import fit_and_export, load_training_data

ROOT = Path(__file__).resolve().parents[2]


def run(data_path=None, seed: int = 42, output_dir=None):
    frame, features = load_training_data(data_path) if data_path else load_training_data()
    specs = [
        ("logistic_regression", LogisticRegression(C=1.0, max_iter=1000,
          class_weight="balanced", solver="lbfgs", random_state=seed), True),
        ("random_forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
          class_weight="balanced_subsample", n_jobs=-1, random_state=seed), False),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=200,
          learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
          random_state=seed, class_weight="balanced"), False),
    ]
    records = []
    for name, estimator, scale in specs:
        preprocessor = None
        # CV evaluates the same preprocessing choice that will be used at refit.
        if scale:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            estimator = make_pipeline(StandardScaler(), estimator)
        result = evaluate_model(estimator, name, frame, features, random_seed=seed)
        records.append(result)
    comparison = pd.DataFrame(records).sort_values("top1_accuracy", ascending=False)
    metrics_path = ROOT / "results" / "metrics" / "model_comparison.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_path, index=False)
    winner = comparison.iloc[0]["model_name"]
    chosen, scale = next((est, sc) for name, est, sc in specs if name == winner)
    out_dir = Path(output_dir) if output_dir else ROOT / "models"
    fit_and_export(chosen, winner, frame, features, scale=scale, random_seed=seed,
                   metrics={"cv_macro_f1": float(comparison.iloc[0]["macro_f1"])},
                   output_dir=out_dir)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(run(args.data, args.seed, args.output_dir).to_string(index=False))
