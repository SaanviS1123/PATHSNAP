# Member 2: Model comparison and Bayesian optimisation

## 3.1 Overall model comparison

The feature table has one row per candidate road arc. The target `is_ref` is binary for known labels; observations with unknown reference labels (`is_ref = -1`) were excluded from classifier fitting and candidate-level scoring. Cross-validation used the frozen five-fold trip assignment in `configs/folds.json`, so rows from a trip never appeared in both sides of a fold. Observation-level Top-1 selected the highest-scored candidate per known observation and compared it with the aligned reference arc, including observations whose reference arc was absent from the candidate set.

The CV columns below are means over the five trip folds. Student-test metrics are a separate development-stage check over ten trips. Route continuity is marked N/A because Member 3's `route_metrics.py` handoff was not available for these runs; no continuity result is inferred from candidate accuracy.

| Model | Algorithm family | Main configuration | CV Top-1 | CV Macro-F1 | Student Top-1 | Student Macro-F1 | Student log loss | Student inference (s) | Route continuity |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression | Linear probabilistic | StandardScaler; C=1; balanced classes; lbfgs; max_iter=1000; seed 42 | 0.7613 | 0.7507 | 0.7950 | 0.7505 | 0.2345 | 0.061 | N/A, C1 pending |
| Random Forest | Tree ensemble | 100 trees; depth 18; min leaf 5; bootstrap sample 0.1; balanced_subsample; seed 42 | 0.7725 | 0.8591 | 0.8024 | 0.8673 | 0.1186 | 1.516 | N/A, C1 pending |
| HistGradientBoosting | Boosting ensemble | 80 iterations; learning rate 0.1; 15 leaves; L2=1; balanced classes; seed 42 | 0.7708 | 0.8220 | 0.8043 | 0.8293 | 0.1717 | 0.294 | N/A, C1 pending |
| KNN | Instance-based | 15 neighbors; distance weights; kd-tree; leaf size 40; 120k training-row cap | 0.7513 | 0.8539 | 0.7856 | 0.8720 | 0.2116 | 109.217 | N/A, C1 pending |
| MLP | Neural network | 32/16 hidden units; Adam; alpha 0.001; batch 512; 40 max iterations; 120k training-row cap | 0.7646 | 0.8648 | 0.7977 | 0.8845 | 0.0875 | 0.596 | N/A, C1 pending |
| Random Forest (TPE tuned) | Tree ensemble | 100 trees; depth 18; min leaf 2; max_features 0.5; bootstrap sample 0.1 | **0.7743** | 0.8666 | **0.8056** | 0.8753 | 0.1054 | 1.411 | N/A, C1 pending |

The six rows represent five distinct model families; the last row is a tuned configuration of Random Forest, not a sixth family. KNN and MLP were trained with the 120,000-row cap recorded in their model cards. The three Member 2 base models and the tuned Random Forest used all known-label training candidates. The common candidate miss rate was 0.00954 on train and 0.00026 on student-test; it is a candidate-generation property, not a classifier difference.

The selected model is the tuned Random Forest. It had the highest mean CV Top-1, and its student-test Top-1 was also the highest of the compared models. MLP had the highest student Macro-F1 (0.8845) and lowest student log loss (0.0875), while its CV Top-1 was lower than tuned RF. KNN's student inference time was much higher than the other models (109.2 s for the measured student-test batch), which is a practical limitation for the API. The complete metrics, fold results, confusion matrices, configurations, and plots are stored under `results/metrics/` and `results/plots/`.

### Logistic Regression

Logistic Regression was included as a fast probabilistic baseline. StandardScaler was fitted inside each CV training fold and then applied to that fold's validation rows. The classifier used balanced class weights, C=1, lbfgs, a 1,000-iteration limit, and seed 42. Its objective was binary log loss. It achieved CV Top-1 0.7613 and Macro-F1 0.7507; on student-test it reached Top-1 0.7950, Macro-F1 0.7505, and log loss 0.2345. Student precision was 0.4078 and recall was 0.9421, showing that the 0.5 threshold favored positive-candidate recall over precision. Inference took 0.061 s on the measured student-test batch.

The similar CV and student Macro-F1 and the improved student Top-1 do not show a clear generalisation drop. They do not prove absence of overfitting: the run did not record a separate training-set score for a train-versus-validation gap.

### Random Forest

Random Forest was chosen to model nonlinear interactions between candidate distance, heading, speed, and local connectivity without requiring feature scaling. The initial configuration used 100 trees, maximum depth 18, minimum leaf size 5, 10% bootstrap samples per tree, balanced-sub-sample class weights, two worker threads, and seed 42. Trees use Gini impurity for their splits; positive-class probability is the forest's averaged class vote. It achieved CV Top-1 0.7725 and Macro-F1 0.8591; student-test Top-1 was 0.8024, Macro-F1 0.8673, and log loss 0.1186. Student precision/recall were 0.6581/0.9006. Student inference took 1.516 s.

The student metrics were not lower than the CV means, so these results do not show a generalisation drop. A separate training score was not retained for this model, so an overfit diagnosis based on a train/validation gap is unavailable.

### HistGradientBoosting

Histogram Gradient Boosting was included as a sequential boosting family with nonlinear decision boundaries. It used 80 boosting iterations, learning rate 0.1, at most 15 leaves per tree, L2 regularisation 1, balanced class weights, and seed 42. The classifier optimised binary log loss. CV Top-1 was 0.7708 and Macro-F1 0.8220; student-test Top-1 was 0.8043, Macro-F1 0.8293, and log loss 0.1717. Student precision was 0.5520 and recall 0.9445. Student inference took 0.294 s. The actual training-score history is recorded in `results/metrics/model_training_history.csv`; the plots show those recorded values only.

CV and student Macro-F1 were close, and student Top-1 was higher than CV mean, so the measured holdout does not show a generalisation drop. The available training history is not a held-out validation curve; it is not used alone to claim that the model did not overfit.

## 4. Bayesian hyperparameter optimisation

Random Forest was selected for TPE search because it was the strongest untuned Member 2 model by CV Top-1. Optuna's TPE sampler used seed 42 and a five-fold GroupKFold objective: mean observation-level Top-1. The search varied `n_estimators` from 50 to 150 in steps of 25, `max_depth` across 10, 14, 18, or 22, `min_samples_leaf` from 2 to 10, and `max_features` across `sqrt`, 0.5, and 0.8. `max_samples=0.1`, balanced-sub-sample class weights, two workers, and seed 42 were held fixed to bound compute. Ten trials were run; their parameters, scores, fold metrics, and costs are in `results/metrics/tuning_trials.csv`.

The best trial was trial 3 (zero-based): 100 trees, depth 18, minimum leaf size 2, and `max_features=0.5`. Total recorded search compute time was 4,028.4 s (about 67.1 minutes). Compared with the untuned forest on the same frozen folds and student-test trips, tuned RF improved CV Top-1 from 0.7725 to 0.7743 and CV Macro-F1 from 0.8591 to 0.8666. Student Top-1 improved from 0.8024 to 0.8056 and student Macro-F1 from 0.8673 to 0.8753. Student log loss fell from 0.1186 to 0.1054. Student inference time was 1.411 s for tuned RF and 1.516 s for untuned RF. The measured CV mean training time increased from 48.9 s to 80.1 s per fold.

Probability quality was checked with log loss, Brier score, and 10-bin expected calibration error. Tuned RF achieved student Brier score 0.03083 and 10-bin ECE 0.03083. Its CV Brier score was 0.03432 and ECE 0.02986. The HMM should consume the emitted positive-class probabilities using the scorer interface; they are candidate-level binary probabilities, not a categorical distribution normalised over all candidates for an observation.

The exported scorer clips exact zero/one tree-vote probabilities to `[1e-12, 1 - 1e-12]` so log-emission calculations remain finite. No post-hoc calibrator was applied; the raw probabilities were retained with the Brier/ECE measurements recorded above.

The frozen feature schema is `member1-v1`. The exported `models/model.joblib`, `models/preprocessor.joblib`, and `models/model_card.json` were generated by the measured refit. Joblib compression reduced the final model artifact to about 16.4 MB.

Route-continuity comparison is intentionally left N/A until C1 is available. This report draft used AI assistance for code and wording; the experiment metrics are taken from the committed run outputs and must be reviewed and explained by the team before final PDF assembly.
