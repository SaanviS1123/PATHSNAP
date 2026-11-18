"""Shared training and artifact export helpers for candidate classifiers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"
MODEL_DIR = ROOT / "models"
REQUIRED = {"trip_id", "obs_idx", "arc_id", "is_ref"}


class CandidateScorer:
    """Stable scorer contract: predict_proba returns P(is_ref=1) as shape (n,)."""

    def __init__(self, estimator, preprocessor, feature_columns):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.feature_columns = list(feature_columns)

    def predict_proba(self, X):
        X = X.loc[:, self.feature_columns]
        if self.preprocessor != "passthrough":
            X = self.preprocessor.transform(X)
        probs = self.estimator.predict_proba(X)
        return probs[:, list(self.estimator.classes_).index(1)]


def load_training_data(path: str | Path = TRAIN_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}. Member 1 must generate train.parquet first.")
    frame = pd.read_parquet(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Training table is missing columns: {sorted(missing)}")
    if SCHEMA_PATH.exists():
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        declared = schema.get("features", schema.get("columns", []))
        if declared and isinstance(declared[0], dict):
            features = [x["name"] for x in declared]
        else:
            features = list(declared)
        if not set(features).issubset(frame.columns):
            raise ValueError("Feature schema columns do not match train.parquet")
    else:
        features = [c for c in frame.columns if c not in REQUIRED]
    if not features:
        raise ValueError("No feature columns found")
    return frame, features


def fit_and_export(estimator, name: str, frame: pd.DataFrame, features: list[str],
                   *, scale: bool = False, random_seed: int = 42,
                   metrics: dict | None = None, output_dir: str | Path = MODEL_DIR):
    frame = frame[frame["is_ref"].astype(int) >= 0].copy()
    labels = frame["is_ref"].astype(int)
    if labels.nunique() != 2:
        raise ValueError("Training data must contain both is_ref classes")
    preprocessor = StandardScaler() if scale else "passthrough"
    X = frame[features]
    if scale:
        X = pd.DataFrame(preprocessor.fit_transform(X), columns=features, index=X.index)
    estimator.fit(X, labels)
    if 1 not in estimator.classes_:
        raise ValueError("Trained classifier has no positive class 1")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(CandidateScorer(estimator, preprocessor, features), out / "model.joblib")
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) if SCHEMA_PATH.exists() else {}
    schema_version = schema.get("version", 1)
    existing_schema = SCHEMA_PATH if SCHEMA_PATH.exists() else None
    if existing_schema and existing_schema.resolve() != (out / "feature_schema.json").resolve():
        schema_text = existing_schema.read_text(encoding="utf-8")
    elif existing_schema:
        schema_text = existing_schema.read_text(encoding="utf-8")
    else:
        schema_text = json.dumps({"version": schema_version, "features": features,
                                  "target": "is_ref", "positive_class": 1}, indent=2) + "\n"
    (out / "feature_schema.json").write_text(schema_text, encoding="utf-8")
    card = {
        "model_name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "config": estimator.get_params(deep=True),
        "feature_schema_version": schema_version,
        "feature_columns": features,
        "train_rows": len(frame), "train_trips": int(frame.trip_id.nunique()),
        "metrics": metrics or {},
        "probability_output": "predict_proba class 1; apply preprocessor.joblib first",
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str) + "\n", encoding="utf-8")
    return estimator, preprocessor
