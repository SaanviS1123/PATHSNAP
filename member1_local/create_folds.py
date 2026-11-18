
from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import GroupKFold

TRAIN_ROOT = Path("Student folder") / "train"
OUTPUT = Path("configs/folds.json")

SEED = 42
N_SPLITS = 5


def main():
    trip_ids = sorted(
        folder.name
        for folder in TRAIN_ROOT.iterdir()
        if folder.is_dir()
    )

    if len(trip_ids) < N_SPLITS:
        raise ValueError(
            "Not enough trips for cross-validation"
        )

    # Each trip is one group.
    groups = np.array(trip_ids)

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    folds = []

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(
            X=np.zeros(len(groups)),
            groups=groups
        ),
        start=1
    ):
        train_trips = groups[train_idx].tolist()
        val_trips = groups[val_idx].tolist()

        assert not (
            set(train_trips) & set(val_trips)
        )

        folds.append({
            "fold": fold_number,
            "train_trip_ids": train_trips,
            "validation_trip_ids": val_trips
        })

    config = {
        "strategy": "GroupKFold",
        "n_splits": N_SPLITS,
        "seed": SEED,
        "seed_note": (
            "GroupKFold does not shuffle; seed is "
            "recorded for reproducibility of other steps."
        ),
        "group_column": "trip_id",
        "total_training_trips": len(trip_ids),
        "folds": folds
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT, "w", encoding="utf-8"
    ) as file:
        json.dump(
            config,
            file,
            indent=2
        )

    print("A6 GROUP-AWARE FOLDS")
    print("Total training trips:", len(trip_ids))
    print("Number of folds:", len(folds))

    for fold in folds:
        print(
            f"Fold {fold['fold']}: "
            f"{len(fold['train_trip_ids'])} training, "
            f"{len(fold['validation_trip_ids'])} validation"
        )

    print("\nSaved:", OUTPUT)
    print("No trip leakage between train and validation.")


if __name__ == "__main__":
    main()
