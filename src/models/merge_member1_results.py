"""Merge completed Member 1 KNN/MLP runs into the measured model comparison."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from src.models.run_all import ROOT, _plot_comparison, _sum_confusions, _write_selection_note
from src.evaluation.harness import candidate_miss_rate


def merge():
    comparison_path = ROOT / "results" / "metrics" / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Run python -m src.models.run_all before merging Member 1 results")
    comparison = pd.read_csv(comparison_path)
    cv_logs_path = ROOT / "results" / "metrics" / "experiments.csv"
    cv_logs = pd.read_csv(cv_logs_path) if cv_logs_path.exists() else pd.DataFrame()
    latest_cv = {row.model_name: row for row in cv_logs.itertuples()} if not cv_logs.empty else {}
    train_table = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    observation_labels = pd.read_parquet(ROOT / "data" / "processed" / "obs_labels.parquet")
    full_train_miss_rate = candidate_miss_rate(train_table,
        observation_labels[observation_labels.partition == "train"])
    history_rows = []
    history_path = ROOT / "results" / "metrics" / "model_training_history.csv"
    if history_path.exists():
        existing_history = pd.read_csv(history_path)
        history_rows = existing_history[~existing_history.model.isin(["knn", "mlp"])].to_dict(orient="records")
    hgb_log = latest_cv.get("hist_gradient_boosting")
    if hgb_log is not None and hasattr(hgb_log, "training_history_json"):
        for history in json.loads(hgb_log.training_history_json):
            history_rows.extend({"model": "hist_gradient_boosting", "run": f"fold_{history['fold']}",
                "history_type": history["history_type"], "iteration": i + 1, "value": value}
                for i, value in enumerate(history["values"]))
    for model_name in ("knn", "mlp"):
        experiments = ROOT / "results" / "metrics" / f"{model_name}_experiments.csv"
        if not experiments.exists():
            raise FileNotFoundError(f"Complete Member 1 experiment file is missing: {experiments}")
        data = pd.read_csv(experiments)
        cv = data[data.partition == "cv"]
        holdout = data[data.partition == "student_test"]
        if cv.fold.nunique() != 5 or len(holdout) != 1:
            raise ValueError(f"{model_name} must have all five CV folds and one student-test run")
        model_card = json.loads((ROOT / "models" / "member1" / model_name / "model_card.json").read_text())
        detailed = json.loads((ROOT / "results" / "metrics" / f"{model_name}_experiments.json").read_text())
        record = {
            "model_name": model_name,
            "algorithm_family": "instance-based" if model_name == "knn" else "neural network",
            "cv_precision": cv.precision.mean(), "cv_recall": cv.recall.mean(), "cv_f1": cv.f1.mean(),
            "cv_top1_accuracy": cv.top1_acc.mean(),
            "cv_macro_f1": cv.macro_f1.mean(),
            "cv_log_loss": cv.log_loss.mean(),
            "cv_brier_score": float("nan"), "cv_ece_10": float("nan"),
            "cv_candidate_miss_rate": full_train_miss_rate,
            "cv_train_time_sec": cv.train_time_s.mean(),
            "cv_inference_time_sec": cv.inference_time_s.mean(),
            "cv_route_continuity": float("nan"),
            "config": json.dumps({"model_params": model_card["configuration"],
                "max_train_rows": model_card["max_train_rows"],
                "sampling": model_card["sampling"]}, sort_keys=True),
            "cv_confusion_matrix": _sum_confusions([r["confusion_matrix"] for r in detailed
                if r.get("partition") == "cv"]),
        }
        row = holdout.iloc[0]
        record.update({"student_top1_accuracy": row.top1_acc,
            "student_precision": row.precision, "student_recall": row.recall,
            "student_f1": row.f1,
            "student_macro_f1": row.macro_f1, "student_log_loss": row.log_loss,
            "student_brier_score": float("nan"), "student_ece_10": float("nan"),
            "student_candidate_miss_rate": row.candidate_miss_rate,
            "student_train_time_sec": row.train_time_s,
            "student_inference_time_sec": row.inference_time_s,
            "student_route_continuity": float("nan"),
            "student_confusion_matrix": next((r["confusion_matrix"] for r in detailed
                if r.get("partition") == "student_test"), None)})
        comparison = comparison[comparison.model_name != model_name]
        comparison = pd.concat([comparison, pd.DataFrame([record])], ignore_index=True)
        member_history = ROOT / "results" / "metrics" / f"{model_name}_loss_history.csv"
        if member_history.exists():
            histories = pd.read_csv(member_history)
            history_rows.extend({"model": model_name, "run": h.run, "history_type": "loss",
                                 "iteration": int(h.epoch), "value": float(h.train_loss)}
                                for h in histories.itertuples())
    family_names = {"logistic_regression": "linear probabilistic",
        "random_forest": "random forest ensemble", "random_forest_tuned": "random forest ensemble",
        "hist_gradient_boosting": "boosting ensemble"}
    for index, row in comparison.iterrows():
        name = row.model_name
        if name in family_names:
            comparison.at[index, "algorithm_family"] = family_names[name]
            log = latest_cv.get(name)
            if log is not None:
                fold_metrics = json.loads(log.fold_metrics_json)
                comparison.at[index, "cv_precision"] = sum(x["precision"] for x in fold_metrics) / len(fold_metrics)
                comparison.at[index, "cv_recall"] = sum(x["recall"] for x in fold_metrics) / len(fold_metrics)
                comparison.at[index, "cv_f1"] = sum(x["f1"] for x in fold_metrics) / len(fold_metrics)
    comparison = comparison.sort_values("cv_top1_accuracy", ascending=False).reset_index(drop=True)
    comparison.to_csv(comparison_path, index=False)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(ROOT / "results" / "metrics" / "model_training_history.csv", index=False)
    _plot_comparison(comparison, history_rows)
    winner = comparison.iloc[0].model_name
    _write_selection_note(comparison, winner)
    if winner in {"knn", "mlp"}:
        source = ROOT / "models" / "member1" / winner
        out = ROOT / "models"
        shutil.copy2(source / "scorer.joblib", out / "model.joblib")
        shutil.copy2(source / "preprocessor.joblib", out / "preprocessor.joblib")
        (out / "model_card.json").write_text(json.dumps({
            "model_name": winner, "selected_by": "highest trip-fold mean observation Top-1",
            "cv_top1_accuracy": float(comparison.iloc[0].cv_top1_accuracy),
            "cv_macro_f1": float(comparison.iloc[0].cv_macro_f1),
            "artifact_source": str(source),
        }, indent=2) + "\n", encoding="utf-8")
    return comparison


if __name__ == "__main__":
    print(merge()[["model_name", "cv_top1_accuracy", "cv_macro_f1",
                   "student_top1_accuracy", "student_macro_f1"]].to_string(index=False))
