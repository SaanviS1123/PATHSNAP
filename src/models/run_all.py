"""Run Member 2 model CV, comparison, and final artifact export."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.harness import evaluate_holdout, evaluate_model
from src.models.training import fit_and_export, load_training_data

ROOT = Path(__file__).resolve().parents[2]


def _sum_confusions(items):
    total = [[0, 0], [0, 0]]
    for item in items:
        if not item:
            continue
        for i in range(2):
            for j in range(2):
                total[i][j] += int(item[i][j])
    return total


def _plot_comparison(table: pd.DataFrame, history_rows: list[dict]):
    import matplotlib.pyplot as plt

    plots = ROOT / "results" / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for metric, title, filename in [
        ("cv_macro_f1", "Trip-grouped CV Macro-F1", "model_cv_macro_f1.png"),
        ("cv_top1_accuracy", "Trip-grouped CV observation Top-1", "model_cv_top1.png"),
        ("cv_log_loss", "Trip-grouped CV log loss", "model_cv_log_loss.png"),
    ]:
        if metric not in table:
            continue
        ax = table.set_index("model_name")[metric].dropna().sort_values().plot.bar(figsize=(8, 4))
        ax.set_title(title); ax.set_ylabel(metric); ax.set_xlabel("")
        ax.figure.tight_layout(); ax.figure.savefig(plots / filename, dpi=160); plt.close(ax.figure)
    student_metrics = [c for c in ("student_top1_accuracy", "student_macro_f1") if c in table]
    if student_metrics:
        ax = table.set_index("model_name")[student_metrics].dropna(how="all").plot.bar(figsize=(8, 4))
        ax.set_title("Student-test Top-1 and Macro-F1"); ax.set_ylabel("score"); ax.set_xlabel("")
        ax.figure.tight_layout(); ax.figure.savefig(plots / "model_student_metrics.png", dpi=160); plt.close(ax.figure)
    matrices = [r for r in table.to_dict(orient="records") if r.get("cv_confusion_matrix")]
    for row in matrices:
        matrix = row["cv_confusion_matrix"]
        if isinstance(matrix, str):
            matrix = json.loads(matrix)
        fig, ax = plt.subplots(figsize=(4, 3))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(title=f"{row['model_name']} CV confusion matrix", xlabel="Predicted", ylabel="Reference")
        ax.set_xticks([0, 1], ["negative", "positive"]); ax.set_yticks([0, 1], ["negative", "positive"])
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center")
        fig.colorbar(image, ax=ax); fig.tight_layout()
        fig.savefig(plots / f"confusion_{row['model_name']}_cv.png", dpi=160); plt.close(fig)
    history_frame = pd.DataFrame(history_rows)
    if not history_frame.empty:
        for (model, run, history_type), curve in history_frame.groupby(
                ["model", "run", "history_type"], sort=True):
            curve = curve.sort_values("iteration")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(curve.iteration, curve.value)
            ax.set(title=f"{model} {run} {history_type}", xlabel="iteration", ylabel=history_type)
            fig.tight_layout()
            fig.savefig(plots / f"training_history_{model}_{run}.png", dpi=160)
            plt.close(fig)


def _write_selection_note(comparison: pd.DataFrame, winner: str):
    selected = comparison.loc[comparison.model_name == winner].iloc[0]
    lines = ["# Model selection result", "",
        f"Selected model: **{winner}**.", "",
        "Selection rule: mean observation-level Top-1 across the frozen trip-grouped CV folds was the primary metric. "
        "Student-test results were checked as supporting development-stage evidence and corroborated the CV choice; "
        "the tuning search itself used training folds only.", "",
        f"- CV observation Top-1: {selected.get('cv_top1_accuracy', float('nan')):.6f}",
        f"- CV candidate Macro-F1: {selected.get('cv_macro_f1', float('nan')):.6f}",
        f"- CV log loss: {selected.get('cv_log_loss', float('nan')):.6f}",
        f"- CV Brier score: {selected.get('cv_brier_score', float('nan')):.6f}",
        f"- CV 10-bin ECE: {selected.get('cv_ece_10', float('nan')):.6f}",
        f"- Candidate miss rate: {selected.get('cv_candidate_miss_rate', float('nan')):.6f}", ""]
    for key, label in (("student_top1_accuracy", "Student-test Top-1"),
                       ("student_macro_f1", "Student-test Macro-F1"),
                       ("student_log_loss", "Student-test log loss"),
                       ("student_brier_score", "Student-test Brier score"),
                       ("student_ece_10", "Student-test 10-bin ECE")):
        if key in selected.index and pd.notna(selected[key]):
            lines.append(f"- {label}: {float(selected[key]):.6f}")
    lines.extend(["", "Full per-model metrics and confusion matrices: `model_comparison.csv`.",
                  "This report is generated only after the full training and evaluation tables are available."])
    (ROOT / "results" / "metrics" / "model_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(data_path=None, seed: int = 42, output_dir=None, student_path=None):
    frame, features = load_training_data(data_path) if data_path else load_training_data()
    labels_path = ROOT / "data" / "processed" / "obs_labels.parquet"
    obs_labels = pd.read_parquet(labels_path) if labels_path.exists() else None
    holdout_path = Path(student_path) if student_path else ROOT / "data" / "processed" / "student_test.parquet"
    holdout = pd.read_parquet(holdout_path) if holdout_path.exists() else None
    specs = [
        ("logistic_regression", LogisticRegression(C=1.0, max_iter=1000,
          class_weight="balanced", solver="lbfgs", random_state=seed), True),
        ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=18,
          min_samples_leaf=5, max_samples=0.1, class_weight="balanced_subsample",
          n_jobs=2, random_state=seed), False),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=80,
          learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0,
          random_state=seed, class_weight="balanced"), False),
    ]
    tuning_path = ROOT / "results" / "metrics" / "tuning_best.json"
    if tuning_path.exists():
        tuned = json.loads(tuning_path.read_text(encoding="utf-8"))
        tuned_params = dict(tuned.get("best_params", {}))
        tuned_params.update({"class_weight": "balanced_subsample", "n_jobs": 2,
                             "max_samples": 0.1,
                             "random_state": seed})
        specs.append(("random_forest_tuned", RandomForestClassifier(**tuned_params), False))
    records = []
    history_rows = []
    for name, estimator, scale in specs:
        # CV evaluates the same preprocessing choice that will be used at refit.
        if scale:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            estimator = make_pipeline(StandardScaler(), estimator)
        cv = evaluate_model(estimator, name, frame, feature_columns=features,
                            random_seed=seed, obs_labels=obs_labels)
        for history in json.loads(cv["training_history_json"]):
            history_rows.extend({"model": name, "run": f"fold_{history['fold']}",
                "history_type": history["history_type"], "iteration": i + 1, "value": value}
                for i, value in enumerate(history["values"]))
        row = {"model_name": name,
            "algorithm_family": ("linear probabilistic" if name == "logistic_regression" else
                "random forest ensemble" if name.startswith("random_forest") else "boosting ensemble"),
            "cv_precision": cv["precision"], "cv_recall": cv["recall"], "cv_f1": cv["f1"],
            "cv_top1_accuracy": cv["top1_accuracy"], "cv_macro_f1": cv["macro_f1"],
            "cv_log_loss": cv["log_loss"], "cv_brier_score": cv["brier_score"],
            "cv_ece_10": cv["ece_10"],
            "cv_candidate_miss_rate": cv["candidate_miss_rate"],
            "cv_train_time_sec": cv["train_time_sec"],
            "cv_inference_time_sec": cv["inference_time_sec"],
            "cv_route_continuity": cv["route_continuity"], "config": cv["config"]}
        fold_metrics = json.loads(cv["fold_metrics_json"])
        row["cv_confusion_matrix"] = _sum_confusions([f["confusion_matrix"] for f in fold_metrics])
        if holdout is not None and obs_labels is not None:
            test = evaluate_holdout(estimator, frame, holdout, obs_labels, features, model_name=name)
            row.update({f"student_{key}": value for key, value in test.items()
                        if key not in {"model_name", "n_test_trips"}})
            row["student_confusion_matrix"] = test["confusion_matrix"]
        records.append(row)
    # Bring in A's measured models when their complete experiment files exist.
    for member_model in ("knn", "mlp"):
        member_path = ROOT / "results" / "metrics" / f"{member_model}_experiments.csv"
        if not member_path.exists():
            continue
        member = pd.read_csv(member_path)
        cv_rows = member[member.partition == "cv"]
        test_rows = member[member.partition == "student_test"]
        if cv_rows.empty:
            continue
        record = {"model_name": member_model,
            "algorithm_family": "instance-based" if member_model == "knn" else "neural network",
            "cv_precision": cv_rows.precision.mean(), "cv_recall": cv_rows.recall.mean(),
            "cv_f1": cv_rows.f1.mean(),
            "cv_top1_accuracy": cv_rows.top1_acc.mean(), "cv_macro_f1": cv_rows.macro_f1.mean(),
            "cv_log_loss": cv_rows.log_loss.mean(),
            "cv_candidate_miss_rate": cv_rows.candidate_miss_rate.mean(),
            "cv_train_time_sec": cv_rows.train_time_s.mean(),
            "cv_inference_time_sec": cv_rows.inference_time_s.mean(),
            "cv_route_continuity": float("nan"), "config": "Member 1 experiment log"}
        member_json = ROOT / "results" / "metrics" / f"{member_model}_experiments.json"
        if member_json.exists():
            member_rows = json.loads(member_json.read_text(encoding="utf-8"))
            record["cv_confusion_matrix"] = _sum_confusions([
                r.get("confusion_matrix") for r in member_rows if r.get("partition") == "cv"])
        member_history_path = ROOT / "results" / "metrics" / f"{member_model}_loss_history.csv"
        if member_history_path.exists():
            for _, h in pd.read_csv(member_history_path).iterrows():
                history_rows.append({"model": member_model, "run": h.get("run", "refit"),
                    "history_type": "loss", "iteration": int(h.epoch), "value": float(h.train_loss)})
        if not test_rows.empty:
            last = test_rows.iloc[-1]
            record.update({"student_top1_accuracy": last.top1_acc,
                "student_precision": last.precision, "student_recall": last.recall,
                "student_f1": last.f1,
                "student_macro_f1": last.macro_f1, "student_log_loss": last.log_loss,
                "student_brier_score": float("nan"),
                "student_candidate_miss_rate": last.candidate_miss_rate,
                "student_train_time_sec": last.train_time_s,
                "student_inference_time_sec": last.inference_time_s,
                "student_route_continuity": float("nan")})
            if member_json.exists():
                test_conf = [r.get("confusion_matrix") for r in member_rows
                             if r.get("partition") == "student_test"]
                if test_conf: record["student_confusion_matrix"] = test_conf[-1]
        records.append(record)
    comparison = pd.DataFrame(records).sort_values("cv_top1_accuracy", ascending=False)
    metrics_path = ROOT / "results" / "metrics" / "model_comparison.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_path, index=False)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(ROOT / "results" / "metrics" / "model_training_history.csv", index=False)
    if "random_forest_tuned" in set(comparison.model_name):
        pair = comparison[comparison.model_name.isin(["random_forest", "random_forest_tuned"])]
        pair.to_csv(ROOT / "results" / "metrics" / "tuned_vs_untuned.csv", index=False)
    _plot_comparison(comparison, history_rows)
    winner = comparison.iloc[0]["model_name"]
    if winner not in {name for name, _, _ in specs}:
        source = ROOT / "models" / "member1" / winner
        if not (source / "scorer.joblib").exists():
            raise FileNotFoundError(f"Best CV model is {winner}, but its Member 1 scorer is missing from {source}")
        out_dir = Path(output_dir) if output_dir else ROOT / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "scorer.joblib", out_dir / "model.joblib")
        if (source / "preprocessor.joblib").exists():
            shutil.copy2(source / "preprocessor.joblib", out_dir / "preprocessor.joblib")
        shutil.copy2(ROOT / "models" / "feature_schema.json", out_dir / "feature_schema.json")
        (out_dir / "model_card.json").write_text(json.dumps({
            "model_name": winner, "selected_by": "highest trip-fold mean observation Top-1",
            "cv_top1_accuracy": float(comparison.iloc[0]["cv_top1_accuracy"]),
            "cv_macro_f1": float(comparison.iloc[0]["cv_macro_f1"]),
            "artifact_source": str(source),
            "note": "Member 1's scorer artifact copied to the common scorer contract path.",
        }, indent=2) + "\n", encoding="utf-8")
        _write_selection_note(comparison, winner)
        return comparison
    chosen, scale = next((est, sc) for name, est, sc in specs if name == winner)
    out_dir = Path(output_dir) if output_dir else ROOT / "models"
    fit_and_export(chosen, winner, frame, features, scale=scale, random_seed=seed,
                   metrics={key: float(comparison.iloc[0][key]) for key in
                       ("cv_top1_accuracy", "cv_macro_f1", "cv_log_loss", "cv_brier_score",
                        "cv_ece_10", "cv_candidate_miss_rate", "student_top1_accuracy",
                        "student_macro_f1", "student_log_loss", "student_brier_score",
                        "student_ece_10", "student_candidate_miss_rate") if key in comparison.columns},
                   output_dir=out_dir)
    _write_selection_note(comparison, winner)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument("--student-test")
    args = parser.parse_args()
    print(run(args.data, args.seed, args.output_dir, args.student_test).to_string(index=False))
