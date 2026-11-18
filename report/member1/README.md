# Member 1 handoff

The two model report sections and section 2.2 are in `eda_and_models.md`, generated
from measured results with `python -m src.data_processing.member1_report`.
The final consolidated PDF remains Member 4's assembly task.

## Reproduce

Run from the repository root with Python 3.13. Put the organiser's train and
student-test trip directories under `data/raw/train` and `data/raw/student_test`.
These are local inputs; do not commit them.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-member1.txt
python -m src.data_processing.run_member1
python -m src.models.train_knn
python -m src.models.train_mlp
python -m src.data_processing.member1_report
python -m pytest -q
```

Preparation regenerates all feature tables, independent observation labels,
training-only candidate sweep, fixed five-fold trip assignments, synthetic
fixtures, inventory, quality diagnostics, static plots, and a Folium map.
Model training defaults to **all known-label rows**, with each scaler fitted only
on that fold's training rows. Optional `--max-train-rows 120000` enables an
explicitly recorded proportional trip/class sample for a smaller machine.
The recorded run's model cards say which setting was actually used.

`notebooks/01_eda.ipynb` reads these measured outputs and can rerun preparation.
No evaluator partition is read. Strict loaders fail with the offending file if
coordinates, column counts, IDs, or finite values are invalid. The quality report
therefore describes successfully validated trips; an invalid input is not silently
cleaned into a different route. Timestamp irregularities are retained and flagged.

## Contracts

- `load_trip(path)` reads `.track`, `.nodes`, `.arcs` only. Offline analysis must
  explicitly request `include_route=True`. IDs are zero-based row indices.
- `project_trip(trip)` returns per-trip metric coordinates, arc geometry, lengths,
  and the exact CRS string. It uses a local azimuthal equidistant projection at the
  circular-mean longitude and mean latitude. `always_xy=True` preserves lon/lat.
  Geodesic checks across all development networks are saved separately.
- `build_features(metric, radius_m, k)` is inference-safe. No reference values,
  route positions, trip IDs, or arc IDs become learned input features.
- `configs/candidates.json` freezes the training-only sweep's chosen radius/K.
  Every observation gets at least one candidate; an empty radius search falls
  back to the nearest arc, retaining ties before deterministic distance/ID capping.
- `data/processed/{train,student_test}.parquet`: one row per
  `(trip_id, obs_idx, arc_id)` plus ordered float32 features and `is_ref`.
  **`is_ref=-1` means unknown and must be excluded from fitting and candidate
  metrics.** Values 0 and 1 are known negative and positive examples.
- `data/processed/obs_labels.parquet`: `trip_id, obs_idx, ref_arc,
  route_position, alignment_distance_m, label_known, partition`.
  `ref_arc=-1` is unknown. This offline table never goes to inference or the UI.
- Candidate miss rate is `known observations without a positive candidate /
  all known observations`. Unknown observations are reported separately.
  Top-1 uses all known observations, including candidate misses.
- `models/feature_schema.json` gives name/dtype/order/version;
  `configs/folds.json` maps all 80 training trip IDs to validation folds.
  No trip is shared between folds or student test.

### Features

| Feature | Meaning / reason |
|---|---|
| `distance_m` | GPS-to-segment distance, the main proximity signal |
| `position_frac` | Clamped projection fraction along the directed arc |
| `heading_diff_rad` | Incoming GPS heading versus directed arc, in [0, pi] |
| `heading_valid` | Heading is usable only with positive time and >1 m motion |
| `speed_mps` | Previous-to-current displacement / positive time gap; otherwise 0 |
| `arc_length_m` | Segment length distinguishes long roads from short junction links |
| `candidate_rank` | One-based order by distance then arc ID |
| `in_degree` | Incoming directed degree of the candidate's start node |
| `out_degree` | Outgoing directed degree of the candidate's end node |
| `time_gap_s` | Actual preceding timestamp interval; 0 for the first point |
| `large_gap` | Preceding interval >5 s or <=0 s (includes the first point) |
| `outside_radius` | Nearest-arc fallback lies beyond the configured search radius |

Heading is unknown at the first point, during <=1 m movement or nonpositive
intervals. It is encoded as angle 0 with `heading_valid=0`. Zero-length arcs use
projection fraction 0. Negative/zero intervals use speed 0 and set `large_gap`.
Geometry features do not require fitting; scaling is confined to each model's
training fold. No scaler is fitted during table preparation.

## Label alignment and uncertainty

A route is an ordered list of arc **occurrences**, including repetitions.
Alignment performs dynamic programming across all occurrences, independent of
candidate generation. Its state can stay or advance, never go backwards in route
order. The per-observation cost is segment distance in metres plus
`15 * (1 - cos(heading_difference))` when heading is valid. Start and end are
unconstrained, since the GPS span may cover only part of a route. An occurrence
may be skipped because a short arc need not receive a GPS observation.
The backtrace assigns one arc per observation; assignments farther than 100 m
are unknown. The 100 m threshold and heading weight are fixed assumptions,
not selected using student-test outcomes.

This is approximate supervision, not independently verified point truth. Arc
progress within one occurrence is not constrained. Loops, parallel roads, noisy
headings, endpoints and broken reference links can remain ambiguous. Distance
and heading also enter model features, so high pointwise scores partly measure
agreement with this alignment procedure. Excluding unknown labels changes the
reported population; unknown counts and independent nearest-reference distances
are retained. Sequence metrics against the original route are Member 3's task.

## Model handoff to Members 2, 3 and 4

B1's harness does not exist yet. To finish A9/A10 without implementing Member 2's
workstream, `src/models/member1_harness.py` provides a reusable runner that uses
the agreed folds/schema. Member 2 can import `run_model()` and merge the exported
KNN/MLP rows into the eventual consolidated five-model comparison. It does not
write into `src/evaluation/`, select a final team model, or perform Bayesian tuning.
Route continuity is explicitly pending C1; it is not replaced with an invented
number. Reevaluate with C1 before final report assembly.

Each model has `model.joblib`, `preprocessor.joblib`, `scorer.joblib` and a model
card under `models/member1/{knn,mlp}/`. Neither overwrites Member 2's final model.
Only load joblib files from trusted sources.

```python
import joblib
from src.features.build import build_features
from src.data_processing.loaders import load_trip
from src.data_processing.geo import project_trip

scorer = joblib.load('models/member1/knn/scorer.joblib')
metric = project_trip(load_trip('data/raw/student_test/00000056'))
# Read radius_m/k from configs/candidates.json in production.
candidates = build_features(metric, radius_m=100, k=16)
candidates['score'] = scorer.predict_proba(candidates)  # shape (n,)
```

The scorer takes columns in schema order, ignores identifiers, checks finite
features, applies its fitted scaler, and returns P(positive), shape `(n,)`.
Probabilities are uncalibrated binary estimates, not a categorical distribution
over a point's candidate set. Member 2/3 must assess calibration and emission
normalisation before using them in HMM decoding.

Results live in `results/metrics/{knn,mlp}_experiments.{csv,json}`, model cards,
recorded MLP training loss, plots and logs. MLP uses no random row-level internal
validation split; this avoids mixing a trip into its own early-stopping validation.
Its maximum-iteration warnings are retained. KNN produces no epoch loss curve.
A <=10k fit-sample diagnostic helps discuss overfitting but is not full training
accuracy; KNN includes its own neighbours and is especially optimistic.

## Remaining team work

Member 2 still owns the shared harness, three other model families, consolidated
comparison, Bayesian tuning and final-model selection. Member 3 owns route
continuity, HMM and sequence metrics. Member 4 owns API/UI, two-laptop integration
and assembly of one final PDF. The team must review and explain this work during
the viva. Code, analysis drafts and report prose were prepared with Codex AI
assistance; every numerical result in this handoff comes from local executed code.
