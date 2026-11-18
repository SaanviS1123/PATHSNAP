# Model-selection policy

Primary: mean observation-level Top-1 across the five frozen trip-grouped folds.
This measures the choice of one road arc per GPS observation. All known labels
count, including absent reference candidates. Unknown alignment labels are
excluded and counted separately.

Ties between model CV Top-1 values use candidate Macro-F1, then model name for a
deterministic order. Tuned parameters replace the winning baseline only on strict
CV Top-1 improvement. Macro-F1, log loss, Brier score, reliability, continuity and
runtime remain recorded diagnostics. Student-test outcomes never choose a family,
search parameters, or calibration. This policy is recorded as a project decision,
not as an invented team meeting or agreement.

Calibration uses a separate training-trip fit/calibration/audit split and accepts
only an increasing sigmoid that improves audit log loss without worse Brier score.
Final student metrics are measured after these choices. Family/search selection
reuse makes the selected CV score optimistic; it is not a nested-CV estimate.
