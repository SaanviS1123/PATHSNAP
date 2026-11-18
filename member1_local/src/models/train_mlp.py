
from pathlib import Path
import json
import random
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(".")
DATA_PATH = ROOT / "data/processed/train.parquet"
SCHEMA_PATH = ROOT / "models/feature_schema.json"
FOLDS_PATH = ROOT / "configs/folds.json"

MODEL_PATH = ROOT / "models/mlp_model.pt"
PREPROCESSOR_PATH = ROOT / "models/mlp_preprocessor.joblib"
METRICS_PATH = ROOT / "results/metrics/mlp_metrics.json"
HISTORY_PATH = ROOT / "results/metrics/mlp_history.csv"
PLOT_PATH = ROOT / "results/plots/mlp_learning_curve.png"

SEED = 42
FOLD_NUMBER = 1
MAX_TRAIN_POSITIVES = 30000
MAX_TRAIN_NEGATIVES = 90000
MAX_VALIDATION_OBSERVATIONS = 10000

EPOCHS = 20
BATCH_SIZE = 2048
LEARNING_RATE = 0.001
PATIENCE = 5


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def sample_training_rows(frame):
    positives = frame[
        frame["is_correct_arc"] == 1
    ]
    negatives = frame[
        frame["is_correct_arc"] == 0
    ]

    positives = positives.sample(
        n=min(len(positives), MAX_TRAIN_POSITIVES),
        random_state=SEED
    )

    negatives = negatives.sample(
        n=min(len(negatives), MAX_TRAIN_NEGATIVES),
        random_state=SEED
    )

    return pd.concat(
        [positives, negatives],
        ignore_index=True
    ).sample(frac=1, random_state=SEED)


def sample_validation_observations(frame):
    observations = frame[
        ["trip_id", "observation_id"]
    ].drop_duplicates()

    if len(observations) > MAX_VALIDATION_OBSERVATIONS:
        observations = observations.sample(
            n=MAX_VALIDATION_OBSERVATIONS,
            random_state=SEED
        )

    return frame.merge(
        observations,
        on=["trip_id", "observation_id"],
        how="inner"
    )


def main():
    started = time.perf_counter()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    schema = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8")
    )
    folds = json.loads(
        FOLDS_PATH.read_text(encoding="utf-8")
    )

    feature_columns = schema["feature_columns"]

    fold = next(
        f for f in folds["folds"]
        if f["fold"] == FOLD_NUMBER
    )

    print("A10 PYTORCH MLP")
    print("Device: CPU")
    print("Validation fold:", FOLD_NUMBER)
    print("Features:", feature_columns)

    columns = [
        "trip_id",
        "observation_id",
        *feature_columns,
        "is_correct_arc"
    ]

    data = pd.read_parquet(
        DATA_PATH,
        columns=columns
    )

    data["trip_id"] = (
        data["trip_id"].astype(str).str.zfill(8)
    )

    train_ids = set(fold["train_trip_ids"])
    val_ids = set(fold["validation_trip_ids"])

    assert train_ids.isdisjoint(val_ids)

    training = data[
        data["trip_id"].isin(train_ids)
    ]
    validation = data[
        data["trip_id"].isin(val_ids)
    ]

    if training.empty or validation.empty:
        raise ValueError(
            "Empty train/validation split."
        )

    train_sample = sample_training_rows(training)
    val_sample = sample_validation_observations(validation)

    print("Training candidate rows:", len(train_sample))
    print("Validation candidate rows:", len(val_sample))

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train = preprocessor.fit_transform(
        train_sample[feature_columns]
    ).astype(np.float32)

    X_val = preprocessor.transform(
        val_sample[feature_columns]
    ).astype(np.float32)

    y_train = train_sample[
        "is_correct_arc"
    ].to_numpy(dtype=np.float32)

    y_val = val_sample[
        "is_correct_arc"
    ].to_numpy(dtype=np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = MLP(len(feature_columns))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("\nTraining MLP...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            train_loss_sum += (
                loss.item() * len(xb)
            )

        train_loss = (
            train_loss_sum / len(train_sample)
        )

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_sum += (
                    loss.item() * len(xb)
                )

        val_loss = (
            val_loss_sum / len(val_sample)
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f}",
            flush=True
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(
            torch.from_numpy(X_val)
        )
        probabilities = (
            torch.sigmoid(logits)
            .numpy()
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    scored = val_sample[
        ["trip_id", "observation_id", "is_correct_arc"]
    ].copy()

    scored["probability"] = probabilities

    best_indices = scored.groupby(
        ["trip_id", "observation_id"]
    )["probability"].idxmax()

    selected = scored.loc[best_indices]

    metrics = {
        "model": "PyTorch MLP",
        "fold": FOLD_NUMBER,
        "seed": SEED,
        "training_candidate_rows": len(train_sample),
        "validation_candidate_rows": len(val_sample),
        "validation_observations": len(selected),
        "validation_positive_labels": int(y_val.sum()),
        "best_validation_loss": float(best_val_loss),
        "epochs_trained": len(history),
        "precision": float(
            precision_score(
                y_val, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_val, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_val, predictions, zero_division=0
            )
        ),
        "candidate_accuracy": float(
            accuracy_score(y_val, predictions)
        ),
        "log_loss": float(
            log_loss(
                y_val,
                np.clip(probabilities, 1e-7, 1 - 1e-7),
                labels=[0, 1]
            )
        ),
        "gps_top1_accuracy": float(
            selected["is_correct_arc"].mean()
        ),
        "elapsed_seconds": round(
            time.perf_counter() - started, 2
        )
    }

    MODEL_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    METRICS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    PLOT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_columns),
            "feature_columns": feature_columns,
            "schema_version": schema["schema_version"]
        },
        MODEL_PATH
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training loss"
    )
    ax.plot(
        history_df["epoch"],
        history_df["validation_loss"],
        label="Validation loss"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("PathSnap MLP Learning Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    print("\nMLP VALIDATION METRICS")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nSaved model:", MODEL_PATH)
    print("Saved preprocessor:", PREPROCESSOR_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved history:", HISTORY_PATH)
    print("Saved plot:", PLOT_PATH)
    print("\nA10 MLP TRAINING COMPLETE")


if __name__ == "__main__":
    main()
