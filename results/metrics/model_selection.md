# Model selection result

Selected model: **random_forest_tuned**.

Selection rule: mean observation-level Top-1 across the frozen trip-grouped CV folds was the primary metric. Student-test results were checked as supporting development-stage evidence and corroborated the CV choice; the tuning search itself used training folds only.

- CV observation Top-1: 0.774335
- CV candidate Macro-F1: 0.866640
- CV log loss: 0.124376
- CV Brier score: 0.034317
- CV 10-bin ECE: 0.029864
- Candidate miss rate: 0.009538

- Student-test Top-1: 0.805621
- Student-test Macro-F1: 0.875261
- Student-test log loss: 0.105383
- Student-test Brier score: 0.030827
- Student-test 10-bin ECE: 0.030827

Full per-model metrics and confusion matrices: `model_comparison.csv`.
This report is generated only after the full training and evaluation tables are available.
