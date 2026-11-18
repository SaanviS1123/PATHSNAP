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
| D016 | 2026-11-18 | Preserve completed experiments and resume final assembly after normalising reserved CSV fields. | Reporting-only `model_name`/`partition` collision; model fits, ten trials and confirmation remain valid under matching input signatures. |
| D017 | 2026-11-18 | Select `hist_gradient_boosting_tuned` with calibration enabled (parameter record 9d686b2c). | Baseline CV winner `hist_gradient_boosting`; TPE best trial 3; evidence: `model_selection.json`, `tuned_vs_untuned.csv`, `calibration_audit.json`. |
| D018 | 2026-11-18 | Restrict pytest discovery to the canonical `tests/` directory. | The separate `member1_local` project shadowed `src` during collection; its scripts require unrelated local fixtures. See `pytest.ini` and `member2_collection_issue.log`. |
| D019 | 2026-11-18 | Complete Member 3 on `main` and preserve existing model artifacts. | User instruction; HMM code lives under `src/hmm/`. |
| D020 | 2026-11-18 | Adopt Member 2’s same-arc/direct head-to-tail continuity definition for C1. Add LCS-based ordered arc-occurrence sequence metrics. | Keeps prior classifier results comparable; no set-only route score. |
| D021 | 2026-11-18 | Use binary scorer probabilities as normalised per-point emission potentials and directed route-distance mismatch transitions. | Log-space Viterbi; transition distance includes partial-arc positions. |
| D022 | 2026-11-18 | Use bounded cached Dijkstra searches; keep parallel directed arcs distinct. | Avoid graph-matrix duplicate-edge summation and unbounded global path searches. |
| D023 | 2026-11-18 | Restart at large/nonpositive time gaps, spatial jumps, empty candidates, or when no feasible state path remains. | Unreachable transitions have zero weight; no fabricated connecting geometry. |
| D024 | 2026-11-18 | Allow at most 20 m backward movement within one arc as GPS jitter. | Larger reverse movement requires a valid directed loop. |
| D025 | 2026-11-18 | Tune decoder parameters on 16 held-out training trips using a separate 48-trip base fit and 16-trip calibration split. | Avoids tuning against in-sample final-model predictions; freeze before student-test evaluation. |
| D026 | 2026-11-18 | Compare classifier-only, classifier plus path bridging, and HMM plus bridging. | Separates Viterbi pointwise gains from filling intermediate road arcs. |
| D027 | 2026-11-18 | Freeze beta 10 m, restart gap 30 s, 16 candidates and 20 m backward-jitter tolerance. | Best of 12 training-only configurations over 16 tuning trips; `configs/hmm.json` and `hmm_tuning.csv`. |
| D028 | 2026-11-18 | Verify organiser file checksums before the offline batch and retain source/data/model fingerprints for decoder caches. | Inference stays reference-free; reference hashing and comparison belong to offline evaluation. |
| D029 | 2026-11-18 | Keep reference-route layers explicit and offline-only; render all ten Folium maps in an isolated headless browser. | All pages loaded tiles without JavaScript errors; screenshot focus is recorded in `folium_render_checks.json`. |
| D030 | 2026-11-18 | Deliver the measured HMM result with the failure case retained. | Student point accuracy 80.36% → 88.46%; direct continuity 90.38% → 97.92%; one trip worsened. All 41 project tests passed. |
