# Primary model-selection metric

**Proposed primary metric: observation-level Top-1 accuracy, averaged across the frozen trip-grouped validation folds.** At inference, the system must select one road arc for each GPS observation, so this metric directly measures the decision the user sees. A candidate-level score can look strong while still ranking the correct arc second within many observations.

Use candidate-level Macro-F1 and log loss as required secondary metrics. Macro-F1 exposes missed positive candidates despite the large number of negative candidate rows; log loss checks probability quality because the HMM consumes scorer probabilities. Candidate miss rate is a feature-generation ceiling and must be reported separately. Route continuity is a sequence-level diagnostic from Member 3, not a substitute for point accuracy.

Select only from held-out trip folds, using the same frozen `configs/folds.json` split for every model. Student-test results are a final held-out check and must not be used to tune the selection decision. The team should confirm this proposed primary metric at the P0/P1 sync; no agreement record or fold configuration is present in this checkout yet.
