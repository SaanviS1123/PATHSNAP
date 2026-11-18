# Member 1 — Local Implementation (A1–A12)

This directory preserves the independent five-feature baseline implementation developed for PathSnap.

It is separate from the team's main 12-feature pipeline and does not replace or modify the existing `src/` modules.

## Completed work

- A1–A3: Data loaders, dataset quality analysis, GPS projection
- A4–A6: Pseudo-label generation, road candidates, feature schema and grouped folds
- A7–A8: Training/test feature generation and exploratory analysis
- A9–A10: KNN and PyTorch MLP baseline training
- A11: Candidate coverage diagnostics and feature validation
- A12: Experiment report

## Dataset setup

The organiser-provided dataset is not included in Git.

Place it at:

    member1_local/Student folder/
        train/
        student_test/

Each trip directory contains `.track`, `.nodes`, `.arcs`, and `.route` files, as provided by the organiser.

## Running the experiments

Open a terminal at the repository root, then run:

    cd member1_local

Install the required Python dependencies in your environment, including NumPy, pandas, SciPy, scikit-learn, PyArrow, joblib, PyTorch, and Matplotlib.

The scripts below use relative paths and should be run from `member1_local`:

    python create_folds.py
    python build_all_labels.py
    python build_all_features.py
    python -m src.models.train_knn
    python -m src.models.train_mlp

These are the intended execution commands; they have not been retested from this copied directory.

## Features

The independent baseline uses five features:

1. distance_m
2. segment_length_m
3. bearing_difference
4. speed_mps
5. time_delta_s

The team's main pipeline uses a different feature schema. Models and preprocessing artifacts should not be interchanged between the two implementations.

## Results and artifacts

- `docs/member1_report.md` — detailed experiment report
- `notebooks/01_eda.ipynb` — exploratory analysis
- `results/metrics/` — evaluation metrics and diagnostics
- `results/plots/` — visualizations
- `models/` — feature schema and trained baseline models
- `configs/` — grouped validation folds
- `tests/` — test fixtures

The baseline results use heuristic pseudo-labels and a sampled first validation fold. They should not be interpreted as final five-fold evaluation results.

## Important notes

Raw organiser datasets and generated feature tables are excluded from Git.

Some scripts are exploratory utilities and may require their own input files or a specific execution order.
