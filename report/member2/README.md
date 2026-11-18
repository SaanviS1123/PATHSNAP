# Member 2 handoff

Member 2 owns the shared evaluation harness, Logistic Regression, Random Forest,
histogram boosting, five-family comparison, Bayesian tuning, final scorer and
report source. Work is on `main`. `DECISIONS.md` records decisions and evidence,
without internal reasoning.

## Reproduce

Use Python 3.13 from the repository root. Member 1 preparation and its complete
KNN/MLP runs must already exist. Only train and student-test data are used.

```bash
python -m pip install -r requirements-member2.txt
python -m src.models.run_all
python -m pytest -q
```

Pytest discovers the canonical `tests/` directory via `pytest.ini`; separate
local projects do not participate in this repository’s test collection. Model
artifact smoke tests use committed fixtures; full-data checks require local
processed data.

The default run completes all five-family comparisons, ten seeded TPE trials,
calibration audit and final export. Search fitting uses an approximately
250,000-row budget per fold, preserving all training trips through proportional
trip/class sampling; validation folds are complete. The winning parameters then
receive full-row five-fold confirmation, so tuned/untuned comparison and final
refits use the same full training population. `--baseline-only` stops after comparison;
`--trials 10` explicitly controls the total study size. At least four trials are
required so the three startup trials are followed by Bayesian proposals.
Matching completed model runs are reused; source/data/schema/fold/config changes
invalidate the relevant signatures. Trials persist in SQLite with sampler state
under `data/processed/member2/`. Actual trial records and every fold's metrics are
saved under `results/metrics/`, including warnings and timings. Baseline KNN/MLP
results are imported only after their recorded input fingerprints match.

The full comparison is `results/metrics/model_comparison.csv`. Per-fold results
are in `experiments.csv`, model JSON files and per-trial JSON files. Tuning has
`tuning_trials.csv`, `tuning_best.json` and `tuned_vs_untuned.csv`. Calibration has
`calibration_audit.json`, held-out reliability plots and final metrics in
`model_selection.json`. `models_and_tuning.md` is the generated report source for
Member 4's single consolidated PDF, not a separate final submission.

## Shared evaluation

`src/evaluation/harness.py` accepts any cloneable sklearn probability estimator,
including pipelines. It uses `configs/folds.json`, fits scalers inside training
folds, excludes `is_ref=-1` only from fitting/candidate metrics, and scores every
observation. Top-1 includes known reference arcs missing from candidate sets.
Tied probabilities choose the lower arc ID. Train/holdout trip overlap is rejected.
All learned input features come from the frozen schema, not identifier/label
columns. Boosting's internal row-level early stopping is disabled.

The primary selection metric is mean fold observation Top-1. Macro-F1 and log loss
are secondary. Student-test values do not select a family, parameters, or whether
to calibrate. Reusing the same CV for family/search selection makes the selected
CV score optimistic; this is explicitly not nested CV.

Classifier continuity is implemented in B's `src/evaluation/continuity.py` while
C1 is absent: consecutive observation predictions must use the same arc or
head-to-tail directed arcs. Unknown labels and large gaps are retained; missing
observation indexes break continuity. Networks are checked against Member 1's
manifest. It reads `.arcs`, never `.route`. This diagnostic is not an HMM, does not
bridge skipped road arcs, and does not replace C's eventual sequence metrics.
Historical Member 1 CV continuity/Brier/ECE are unavailable because those OOF
probabilities were not saved; student diagnostics are measured for every model.

## Inference contract

```python
from src.models.training import load_scorer

scorer = load_scorer('models')
probability = scorer.predict_proba(candidate_features)  # shape (n,)
```

`load_scorer` checks artifact hashes and schema version before loading trusted
local joblib artifacts. The returned scorer takes **raw** schema features, orders
the columns, checks finiteness, applies preprocessing once, optionally applies
sigmoid calibration, and returns finite positive probabilities. Empty batches
return an empty array. No reference routes or labels are read at inference.

`model.joblib` already includes preprocessing/calibration. The separately exported
`preprocessor.joblib` is for audit/reconstruction; applying it before the scorer
would preprocess twice. Binary probabilities are not a categorical distribution
over an observation's candidates; Member 3 must choose and document emission
normalisation. Model artifacts are checked against the 100 MB per-file limit.
Experimental model exports are under `models/member2/`; the chosen final scorer
and its schema/card are at the top level of `models/`. Member 1 exports are retained.

Calibration is considered using 48/16/16 disjoint training-trip fit/calibration/
audit sets. It is accepted only if monotonic and audit log loss improves without
worse Brier score. If accepted, a final sigmoid is fitted to training OOF scores;
the selected base model is fitted on all 80 training trips. Student results are
reported only after this decision. Large training/audit population differences
and the surrogate label-alignment uncertainty still limit interpretation.

Member 3 owns HMM/Viterbi, final continuity definitions and sequence evaluation.
Member 4 owns API/UI integration and final PDF assembly. The source and reports
disclose Codex AI assistance; numerical claims come from executed local runs.
