# PathSnap — Member 1 Technical Report

## 2.2 Data Processing, Candidate Generation, and Feature Engineering

### 2.2.1 Dataset Overview

PathSnap addresses map matching and route reconstruction by associating noisy GPS observations with candidate road-network segments.

The dataset consists of 80 training trips and 10 student-test trips. Each trip contains GPS observations and road-network information. Training trips additionally provide reference route information used to construct provisional labels.

| Dataset statistic | Training | Student Test |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Generated candidate rows | 3,248,982 | 366,616 |

The data-loading pipeline supports four file formats:

- `.track`: GPS longitude, latitude, and timestamp observations.
- `.nodes`: Road-network node coordinates.
- `.arcs`: Road-network connections between nodes.
- `.route`: Reference route arc identifiers, used for training alignment.

The inventory and quality checks identified 14 disconnected reference-route links across the 90 trips. No isolated nodes or timestamp problems were reported.

### 2.2.2 Coordinate Projection

GPS and road-network coordinates were converted from longitude and latitude to a local metric coordinate system using a per-trip equirectangular projection.

This allows road-segment distances and lengths to be calculated in meters rather than angular degrees.

### 2.2.3 Pseudo-Label Generation

Reference route arcs were aligned to GPS observations to generate provisional candidate labels.

The alignment procedure uses spatial proximity and a forward-progression preference along the reference route. These labels are heuristic and should not be interpreted as independently verified ground truth.

All 208,417 training GPS observations received a provisional arc label. The median alignment distance was approximately 5.69 meters, although some difficult trips showed larger discrepancies.

### 2.2.4 Candidate Generation

A spatial index was built over road-segment midpoints. Nearby segments were retrieved and ranked using exact point-to-segment distances.

The selected baseline configuration was:

| Parameter | Value |
|---|---:|
| Initial spatial search | 500 segments |
| Maximum candidates per observation | 30 |
| Search radius | 100 meters |
| Empty-radius behavior | Nearest-segment fallback |

The candidate sweep achieved 97.69% provisional-label coverage on its sampled evaluation set using this configuration.

The full training feature dataset achieved 90.04% coverage, illustrating that performance on a sample does not necessarily represent performance over all observations.

### 2.2.5 Feature Engineering

Five numerical features were constructed for each GPS observation and candidate road segment.

| Feature | Description | Unit |
|---|---|---|
| `distance_m` | GPS point-to-segment distance | meters |
| `segment_length_m` | Length of candidate road segment | meters |
| `bearing_difference` | Difference between GPS movement bearing and road-segment orientation | degrees |
| `speed_mps` | Estimated GPS movement speed | meters/second |
| `time_delta_s` | Elapsed time between GPS observations | seconds |

Training rows contain the binary target `is_correct_arc`, indicating whether a candidate arc matches the provisional reference label.

Student-test rows do not contain this target.

### 2.2.6 Data Validation

The generated training and test datasets passed the following checks:

- No duplicate `(trip_id, observation_id, arc_id)` candidate rows.
- No missing feature values.
- At least one candidate for every GPS observation.
- At most one positive candidate per training observation.
- No training labels in the student-test dataset.

A five-fold GroupKFold configuration was created using trip identifiers to prevent observations from the same trip appearing in both training and validation within a fold.

## KNN Baseline

A K-Nearest Neighbors classifier was trained using the five standardized numerical features.

The initial experiment used Fold 1, a reproducible training sample of 120,000 candidate rows, and 10,000 complete validation GPS observations.

The model used 15 neighbors with distance-based weighting.

### KNN Results

| Metric | Value |
|---|---:|
| Precision | 0.4966 |
| Recall | 0.8464 |
| F1 score | 0.6259 |
| Candidate-level accuracy | 95.02% |
| Log loss | 0.1787 |
| GPS top-1 accuracy | 51.28% |
| Training and evaluation time | 9.66 seconds |

GPS top-1 accuracy measures whether the highest-scoring candidate for each GPS observation matches its provisional label.

Candidate-level accuracy must be interpreted cautiously because most candidate rows are negative.

## MLP Baseline

A feedforward neural network was implemented using PyTorch on CPU.

The architecture consisted of:

- Five input features.
- A hidden layer of 64 neurons with ReLU activation.
- A hidden layer of 32 neurons with ReLU activation.
- A single binary classification output.

The network used Adam optimization, binary cross-entropy with logits, and early stopping.

The model trained for 10 epochs, with the best validation loss occurring at epoch 5.

### MLP Results

| Metric | Value |
|---|---:|
| Precision | 0.5133 |
| Recall | 0.8474 |
| F1 score | 0.6394 |
| Candidate-level accuracy | 95.29% |
| Log loss | 0.1188 |
| GPS top-1 accuracy | 54.10% |
| Training and evaluation time | 28.62 seconds |

### KNN versus MLP

On the same Fold 1 validation sample, MLP improved GPS top-1 accuracy from 51.28% to 54.10%, a gain of 2.82 percentage points.

MLP also achieved a higher F1 score and lower log loss, although it required more computation than KNN.

These results are preliminary single-fold evaluations rather than complete five-fold cross-validation results.

## Exploratory Data Analysis

EDA examined trip sizes, candidate coverage, candidate distances, and geographical route alignment.

The median training trip contained approximately 1,529 GPS observations.

The median candidate-to-GPS distance across generated candidate rows was approximately 45.75 meters.

Three interactive maps were generated for Trips 1, 33, and 54 to compare GPS trajectories against reference routes.

The EDA outputs include:

- `results/plots/eda_trip_sizes.png`
- `results/plots/eda_candidate_coverage.png`
- `results/plots/eda_candidate_distances.png`
- `results/plots/mlp_learning_curve.png`
- `maps/eda_trip_00000001.html`
- `maps/eda_trip_00000033.html`
- `maps/eda_trip_00000054.html`
- `notebooks/01_eda.ipynb`

## Candidate-Miss Analysis

The full training dataset contains 208,417 GPS observations.

Of these, 187,658 observations have a candidate matching the provisional reference label, while 20,759 do not.

| Measure | Value |
|---|---:|
| Provisional-label candidate coverage | 90.04% |
| Candidate miss rate | 9.96% |
| Missing positive candidates | 20,759 |
| Share of misses from Trips 33 and 54 | 93.89% |

Trip 54 contributed 16,355 misses, and Trip 33 contributed 3,135 misses.

These two trips therefore account for most of the observed candidate-coverage problem.

On the sampled Fold 1 validation set, only 6,998 of 10,000 observations had a positive candidate, limiting the maximum possible GPS top-1 accuracy to 69.98% under the provisional labels.

## Limitations and Future Work

The principal limitations are:

1. Pseudo-labels are generated heuristically and may contain alignment errors.
2. Candidate generation can exclude the provisional reference arc, especially for difficult trips.
3. The current KNN and MLP results cover only one of the five predefined validation folds.
4. Candidate-level accuracy can be inflated by class imbalance.
5. The baseline candidate radius and candidate count may be insufficient for sparse or geographically inconsistent trips.

Potential improvements include adaptive candidate radius selection, increasing the candidate count for difficult observations, improved reference-route alignment, temporal sequence decoding, and complete five-fold evaluation.

## Conclusion

Member 1 implemented the dataset loading, quality analysis, coordinate projection, pseudo-labeling, spatial candidate generation, feature engineering, exploratory analysis, and initial KNN and MLP baselines.

The complete training and student-test feature datasets were generated and validated successfully.

The MLP outperformed KNN on the preliminary Fold 1 experiment, while candidate-recall analysis identified Trips 33 and 54 as the main sources of missing positive candidates.

The resulting artifacts provide a reproducible foundation for subsequent map-matching model development and evaluation.
