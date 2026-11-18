"""Trip-grouped evaluation for candidate-level map-matching classifiers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
FOLDS_PATH = ROOT / "configs" / "folds.json"
EXPERIMENTS_PATH = ROOT / "results" / "metrics" / "experiments.csv"
REQUIRED_COLUMNS = {"trip_id", "obs_idx", "arc_id", "is_ref"}


def _index(frame: pd.DataFrame, indexes: Iterable[int]) -> pd.DataFrame:
    return frame.iloc[list(indexes)]


def _load_fold_map(path: Path, groups: np.ndarray) -> dict[str, int] | None:
    """Read an optional frozen trip->fold mapping; accept common JSON layouts."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = raw.get("trip_to_fold", raw.get("folds", raw)) if isinstance(raw, dict) else raw
    if isinstance(mapping, list):
        mapping = {str(trip): int(fold) for fold, trips in enumerate(mapping) for trip in trips}
    if not isinstance(mapping, dict):
        raise ValueError(f"Unsupported fold configuration in {path}")
    normalized = {str(k): int(v) for k, v in mapping.items() if isinstance(v, (int, float))}
    if set(map(str, groups)) - set(normalized):
        raise ValueError("configs/folds.json does not assign every trip in this dataset")
    return normalized


def make_group_splits(groups: Iterable[Any], n_splits: int = 5,
                      folds_path: str | Path | None = None):
    """Yield row indices with no trip leakage, using frozen folds when present."""
    group_values = np.asarray(groups).astype(str)
    path = Path(folds_path) if folds_path else FOLDS_PATH
    fold_map = _load_fold_map(path, group_values)
    unique = np.unique(group_values)
    if fold_map is None:
        if len(unique) < n_splits:
            raise ValueError(f"Need at least {n_splits} distinct trips; found {len(unique)}")
        splitter = GroupKFold(n_splits=n_splits)
        dummy = np.zeros(len(group_values))
        yield from splitter.split(dummy, groups=group_values)
        return
    fold_ids = sorted(set(fold_map[g] for g in unique))
    if len(fold_ids) < 2:
        raise ValueError("Frozen fold assignment must contain at least two folds")
    for fold in fold_ids:
        val = np.flatnonzero(np.isin(group_values, [g for g in unique if fold_map[g] == fold]))
        train = np.flatnonzero(np.isin(group_values, [g for g in unique if fold_map[g] != fold]))
        yield train, val


def _positive_probability(estimator, X) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Estimator must implement predict_proba for HMM-compatible scoring")
    probs = np.asarray(estimator.predict_proba(X))
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if probs.ndim != 2 or 1 not in classes:
        raise ValueError("predict_proba must return a 2D matrix containing positive class 1")
    scores = probs[:, classes.index(1)]
    return np.clip(scores.astype(float), 1e-12, 1 - 1e-12)


def _top1(frame: pd.DataFrame, scores: np.ndarray,
          obs_labels: pd.DataFrame | None = None) -> float:
    work = frame[["trip_id", "obs_idx", "arc_id", "is_ref"]].copy()
    work["_score"] = scores
    winners = work.loc[work.groupby(["trip_id", "obs_idx"], sort=False)["_score"].idxmax()]
    if obs_labels is None:
        known = winners[winners["is_ref"] >= 0]
        return float(known["is_ref"].astype(int).mean()) if len(known) else float("nan")
    labels = obs_labels[["trip_id", "obs_idx", "ref_arc"]].dropna(subset=["ref_arc"])
    labels = labels[labels["ref_arc"] >= 0]
    joined = labels.merge(winners[["trip_id", "obs_idx", "arc_id"]],
                          on=["trip_id", "obs_idx"], how="left", validate="one_to_one")
    return float((joined["arc_id"] == joined["ref_arc"]).mean()) if len(joined) else float("nan")


def _expected_calibration_error(y_true, probability, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(probability, edges[1:-1], right=False), n_bins - 1)
    return float(sum(np.mean(bins == b) * abs(np.mean(probability[bins == b]) -
                np.mean(y_true[bins == b])) for b in range(n_bins) if np.any(bins == b)))


def candidate_miss_rate(candidates: pd.DataFrame, obs_labels: pd.DataFrame | None) -> float:
    """Fraction of known reference observations whose reference arc is not a candidate."""
    if obs_labels is None or obs_labels.empty:
        return float("nan")
    needed = {"trip_id", "obs_idx", "ref_arc"}
    if not needed.issubset(obs_labels.columns):
        raise ValueError(f"obs_labels must contain {sorted(needed)}")
    refs = obs_labels.dropna(subset=["ref_arc"])
    refs = refs[refs["ref_arc"] >= 0]
    refs = refs[refs["trip_id"].astype(str).isin(candidates["trip_id"].astype(str).unique())]
    if refs.empty:
        return float("nan")
    keys = candidates[["trip_id", "obs_idx", "arc_id"]].drop_duplicates()
    joined = refs.merge(keys, left_on=["trip_id", "obs_idx", "ref_arc"],
                        right_on=["trip_id", "obs_idx", "arc_id"], how="left", indicator=True)
    return float((joined["_merge"] == "left_only").mean())


def _append_result(row: dict, output_path: str | Path | None) -> None:
    path = Path(output_path) if output_path else EXPERIMENTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        previous = pd.read_csv(path)
        keys = ["model_name", "config", "random_seed"]
        if set(keys).issubset(previous.columns) and set(keys).issubset(new.columns):
            same = pd.Series(True, index=previous.index)
            for key in keys:
                same &= previous[key].fillna("").astype(str).eq(str(row[key]))
            previous = previous.loc[~same]
        new = pd.concat([previous, new], ignore_index=True)
    new.to_csv(path, index=False)


def evaluate_model(estimator, model_name: str, data: pd.DataFrame,
                   feature_columns: list[str] | None = None,
                   config: dict | None = None, random_seed: int = 42,
                   obs_labels: pd.DataFrame | None = None, n_splits: int = 5,
                   folds_path: str | Path | None = None,
                   continuity_fn: Callable | None = None,
                   output_path: str | Path | None = None,
                   log: bool = True) -> dict:
    """Cross-validate an estimator over trips and report row and observation metrics.

    ``continuity_fn`` is an optional adapter to Member 3's route_metrics module;
    it receives ``(validation_rows, positive_scores)`` and returns a scalar.
    """
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")
    features = feature_columns or [c for c in data.columns if c not in REQUIRED_COLUMNS]
    if not features:
        raise ValueError("No feature columns supplied or found")
    X = data[features]
    y_all = data["is_ref"].astype(int).to_numpy()
    groups = data["trip_id"].astype(str).to_numpy()
    fold_rows = []
    histories = []
    for fold_no, (train_idx, val_idx) in enumerate(make_group_splits(groups, n_splits, folds_path), 1):
        model = clone(estimator)
        train_known = train_idx[y_all[train_idx] >= 0]
        val_known = val_idx[y_all[val_idx] >= 0]
        if not len(train_known) or not len(val_known):
            raise ValueError(f"Fold {fold_no} has no known-label training or validation candidates")
        X_train, X_val = _index(X, train_known), _index(X, val_known)
        y_train, y_val = y_all[train_known], y_all[val_known]
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        if hasattr(model, "loss_curve_"):
            histories.append({"fold": fold_no, "history_type": "loss",
                              "values": np.asarray(model.loss_curve_, dtype=float).tolist()})
        elif hasattr(model, "train_score_"):
            histories.append({"fold": fold_no, "history_type": "train_score",
                              "values": np.asarray(model.train_score_, dtype=float).tolist()})
        start = time.perf_counter()
        scores = _positive_probability(model, X_val)
        infer_seconds = time.perf_counter() - start
        rows = data.iloc[val_known].copy()
        fold_labels = None
        if obs_labels is not None:
            fold_labels = obs_labels[obs_labels.trip_id.astype(str).isin(groups[val_idx])]
        pred = (scores >= 0.5).astype(int)
        row = {
            "fold": fold_no,
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
            "macro_f1": f1_score(y_val, pred, average="macro", zero_division=0),
            "log_loss": log_loss(y_val, scores, labels=[0, 1]),
            "top1_accuracy": _top1(rows, scores, fold_labels),
            "brier_score": brier_score_loss(y_val, scores),
            "ece_10": _expected_calibration_error(y_val, scores),
            "confusion_matrix": confusion_matrix(y_val, pred, labels=[0, 1]).tolist(),
            "train_time_sec": train_seconds,
            "inference_time_sec": infer_seconds,
            "n_train_rows": len(train_idx), "n_validation_rows": len(val_idx),
            "n_validation_trips": rows.trip_id.nunique(),
        }
        row["route_continuity"] = continuity_fn(rows, scores) if continuity_fn else np.nan
        fold_rows.append(row)
    fold_frame = pd.DataFrame(fold_rows)
    summary = {c: float(fold_frame[c].mean()) for c in fold_frame.columns
               if c not in {"fold", "confusion_matrix"}}
    summary.update({
        "model_name": model_name,
        "config": json.dumps(config or estimator.get_params(deep=True), sort_keys=True, default=str),
        "random_seed": int(random_seed),
        "candidate_miss_rate": candidate_miss_rate(data, obs_labels),
        "n_folds": len(fold_frame),
        "fold_metrics_json": fold_frame.to_json(orient="records"),
        "training_history_json": json.dumps(histories),
    })
    if log:
        _append_result(summary, output_path)
    return summary


def evaluate_holdout(estimator, train_data: pd.DataFrame, test_data: pd.DataFrame,
                     obs_labels: pd.DataFrame, feature_columns: list[str],
                     *, model_name: str = "model") -> dict:
    """Fit on all known training candidates and score a disjoint held-out partition."""
    train = train_data[train_data["is_ref"].astype(int) >= 0]
    test = test_data[test_data["is_ref"].astype(int) >= 0]
    if train.empty or test.empty:
        raise ValueError("Training and holdout sets must have known-label candidate rows")
    model = clone(estimator)
    started = time.perf_counter()
    model.fit(train[feature_columns], train["is_ref"].astype(int))
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = _positive_probability(model, test[feature_columns])
    inference_seconds = time.perf_counter() - started
    y = test["is_ref"].astype(int).to_numpy()
    test_labels = obs_labels[obs_labels.trip_id.astype(str).isin(test_data.trip_id.astype(str).unique())]
    row = {
        "model_name": model_name,
        "precision": precision_score(y, scores >= 0.5, zero_division=0),
        "recall": recall_score(y, scores >= 0.5, zero_division=0),
        "f1": f1_score(y, scores >= 0.5, zero_division=0),
        "macro_f1": f1_score(y, scores >= 0.5, labels=[0, 1], average="macro", zero_division=0),
        "log_loss": log_loss(y, scores, labels=[0, 1]),
        "brier_score": brier_score_loss(y, scores),
        "ece_10": _expected_calibration_error(y, scores),
        "confusion_matrix": confusion_matrix(y, scores >= 0.5, labels=[0, 1]).tolist(),
        "top1_accuracy": _top1(test, scores, test_labels),
        "candidate_miss_rate": candidate_miss_rate(test_data, test_labels),
        "train_time_sec": train_seconds,
        "inference_time_sec": inference_seconds,
        "n_test_trips": int(test.trip_id.nunique()),
    }
    return row
