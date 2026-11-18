# Decision log

Concise decisions and outcomes only. Dates use Asia/Kolkata. No internal thought process is recorded.

| ID | Date | Decision | Evidence / consequence |
| --- | --- | --- | --- |
| D001 | 2026-11-18 | Complete Member 2 directly on `main`; preserve Member 1 artifacts. | User instruction; exports for experiments use `models/member2/`. |
| D002 | 2026-11-18 | Select by mean observation Top-1 across the five frozen trip folds; use Macro-F1 and log loss as secondary metrics. | Aligns selection with choosing one road arc per observation. Student-test outcomes do not choose algorithms or parameters. |
| D003 | 2026-11-18 | Exclude `is_ref=-1` from supervised fitting and candidate metrics; retain every observation for inference and continuity. | Member 1 schema; known-label Top-1 includes candidate misses. |
| D004 | 2026-11-18 | Fit preprocessing within each training fold; disable boosting's row-level early stopping. | Prevents the same trip entering an internal training/validation split. |
| D005 | 2026-11-18 | Reuse Member 1 KNN/MLP measurements only after validating their data, schema and fold fingerprints. | Avoids repeating completed experiments; missing historical metrics remain explicitly unavailable. |
| D006 | 2026-11-18 | Measure classifier continuity as consecutive predictions on identical arcs or directed head-to-tail arcs. | Interim B-owned diagnostic while C1 is absent; no HMM or shortest-path bridging is implied. |
| D007 | 2026-11-18 | Use bounded Random Forest depth, tree count and bootstrap size; record exact settings. | Keeps the baseline and tuning feasible on a laptop and constrains export size. |
| D008 | 2026-11-18 | Tune the measured CV winner with seeded Optuna TPE, ten trials and three startup trials. | Ensures Bayesian proposals occur; all five frozen validation folds evaluate each completed trial. |
| D009 | 2026-11-18 | Audit sigmoid calibration using disjoint training-trip fit/calibration/audit groups; student test does not decide calibration. | Compare training-audit log loss and Brier score before enabling calibration. |
| D010 | 2026-11-18 | Export a scorer that accepts raw schema features and applies preprocessing/calibration once. | `model.joblib.predict_proba(X)` returns shape `(n,)`; separate preprocessor is for inspection/reconstruction. |
| D011 | 2026-11-18 | Persist fold metrics, out-of-fold scores, search trials, warnings and input signatures. | Resume only matching runs; report measured results and disclose selection-CV optimism. |
| D012 | 2026-11-18 | Verify road-network checksums before computing continuity. | Reject changed networks; the continuity calculation reads `.arcs`, never `.route`. |
| D013 | 2026-11-18 | Budget TPE training at approximately 250,000 rows per fold, sampled proportionally within every training trip/class; score full validation folds. Confirm the best parameters with full-row five-fold CV before selection. | Measured forest fits take 70–80 s per full fold. Final baseline, confirmation and refit use all known rows. |
| D014 | 2026-11-18 | Keep class-balanced Member 2 baselines and assess their probability prior through calibration diagnostics. | Exact weights are in model configurations; student test does not choose weighting or calibration. |
| D015 | 2026-11-18 | Tune Histogram Boosting under the predeclared primary-metric rule. Treat its baseline lead as a near-tie, not statistical superiority. | CV Top-1: 0.772413 versus Random Forest 0.772384; `model_comparison.csv`. |
