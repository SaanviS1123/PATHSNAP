# PathSnap: Hackathon Work Distribution

Plan for splitting the PathSnap map-matching hackathon across the team. It is derived from the five briefing PDFs in [`statement/pathsnap/`](statement/pathsnap/) and a first look at the data in `data/raw/` (local only, gitignored).

**Who does what, task by task:** see [`TEAM_ASSIGNMENTS.md`](TEAM_ASSIGNMENTS.md).

---

## 1. What we are building

Noisy GPS points go in, and the road segments the vehicle actually drove come out.

```
 .track + .nodes + .arcs                                          ┌─────────────────────────┐
 (per trip)                                                       │ Laptop 1: FastAPI       │
      │                                                           │  models + HMM + network │
      ▼                                                           │  ground truth server-   │
 [A] project to metric CRS ─► candidate arcs per GPS point        │  side only              │
      │                                                           └───────────▲─────────────┘
      ▼                                                                       │ POST /predict {trip_id}
 [A] features per (point, candidate arc)                                      │
      │                                                           ┌───────────┴─────────────┐
      ▼                                                           │ Laptop 2: UI            │
 [B] classifier ─► P(candidate is the correct arc)                │  10 eval trips, map,    │
      │                                                           │  predicted route,       │
      ▼                                                           │  metrics                │
 [C] HMM + Viterbi over candidates, using road connectivity       └─────────────────────────┘
      │
      ▼
 predicted arc per GPS point  ──►  [D] API / UI / report
```

**Working interpretation (confirm in P0):** every trip ships its own road network, so we cannot classify directly over arc IDs. We generate candidate arcs per GPS point and train a **binary candidate scorer** (is this the reference arc?). That is what the briefing's "candidate probabilities", "candidate miss rate" and "observation-level Top-1" imply. The HMM then picks a connected sequence of candidates.

## 2. Dataset facts (first look)

These were measured with a throwaway script. Workstream A must **re-derive them in the team's own EDA** (the report has to be our own work and every number traceable to the repo).

| | Train | Student test | Evaluator test |
|---|---|---|---|
| Trips | 80 | 10 | 10 (not on our disk, final evaluation only) |
| GPS points total | 208,417 | 19,160 | n/a |
| GPS points / trip (min / median / max) | 361 / 1,529 / 19,313 | 262 / 1,420 / 3,958 | n/a |
| Road nodes / trip | 299 / 9,608 / 155,003 | 1,770 / 6,596 / 26,635 | n/a |
| Road arcs / trip | 649 / 19,526 / 307,392 | 3,577 / 13,145 / 56,027 | n/a |
| Route arcs / trip | 59 / 280 / 2,266 | 95 / 250 / 1,323 | hidden |
| Mean share of time steps that are not 1 s | 6.2% | 1.8% | n/a |
| Route arcs chained head-to-tail | 31,243 of 31,256 | 3,466 of 3,467 | n/a |

What this means for the design:

- **Trips are spread across the world** (train longitude −100.4 to 137.2, latitude −34.6 to 61.3). Use a **per-trip local metric CRS** (for example UTM from the track centroid). Never use one global CRS or Euclidean distance in degrees.
- **Each trip has its own road network**, from about 300 to 155k nodes. Spatial indexing and per-trip graphs are mandatory.
- **A route has far fewer arcs than there are GPS points** (trip `00000001`: 525 route arcs vs 1,315 points). The `.route` file is an arc sequence, not a per-point label. Per-point labels need an explicit alignment step (section 6, Risk 1).
- **A few reference-route links are not connected**, so the reference is not perfect ground truth. Document this, as the briefing asks.
- **Some trips are large** (up to 19k points and 307k arcs in train), so decoder speed and API latency matter.

## 3. Roster and workstreams

Four workstreams. Names are placeholders.

| ID | Workstream | Owner | Models owned (report subsections) |
|---|---|---|---|
| **A** | Data, candidates, features, EDA | `<name>` | KNN, MLP |
| **B** | Model experiments, selection, Bayesian tuning, artifacts | `<name>` | Logistic Regression, Random Forest, Boosting (HistGB or LightGBM) |
| **C** | HMM/Viterbi, route metrics, prediction maps | `<name>` | none (owns the decoder comparison) |
| **D** | API, UI, deployment, integration, README, report assembly | `<name>` | none |

The five-model minimum is covered by six candidates: Logistic Regression (linear), KNN (instance-based), Random Forest (ensemble), Boosting, MLP (neural), plus one spare (Gaussian NB or calibrated SVM) so a failed model does not drop us below five. B is the heaviest stream, so A takes KNN and MLP once the feature tables are stable.

**Not four people?**
- **3 people:** C also owns the API. A also owns the UI. D's other duties go to B.
- **5 people:** split B into "model experiments" and "tuning plus artifacts".

**Viva rule:** each person writes and explains the report sections for their own work, and must also be able to explain the other stages at overview level. Schedule one cross-teach session before the final evaluation (P5).

## 4. Interface contracts

These let all four people work in parallel. **Freeze them at gate G1, and change one only by telling everyone.**

### 4.1 Candidate feature table (A produces, B and C consume)

Parquet files: `data/processed/train.parquet`, `data/processed/student_test.parquet` (regenerable, so gitignored). One row per `(trip_id, obs_idx, arc_id)`.

| Column | Meaning |
|---|---|
| `trip_id` | 8-character zero-padded string, the GroupKFold key |
| `obs_idx` | 0-based GPS row index |
| `arc_id` | 0-based row in `.arcs` |
| `<feature columns>` | listed in `models/feature_schema.json` (name, dtype, order) |
| `is_ref` | 0 or 1 label derived from the route alignment |

Also write `data/processed/obs_labels.parquet` (`trip_id, obs_idx, ref_arc` where known). Candidate miss rate is computed from it.

Fold assignment lives in `configs/folds.json` (GroupKFold by `trip_id`, fixed seed), so B and C evaluate on identical splits.

### 4.2 Scorer (B produces, C and D consume)

```python
scorer.predict_proba(X: pd.DataFrame) -> np.ndarray   # shape (n,), P(is_ref = 1)
```

`models/` holds `model.joblib`, `preprocessor.joblib` (fitted on train only), `feature_schema.json` and `model_card.json` (config, seed, metrics).

### 4.3 Decoder (C produces, D consumes)

```python
decode(cands: pd.DataFrame,      # trip_id, obs_idx, arc_id, score
       network: RoadNetwork,     # directed arc graph for one trip
       params: HMMParams) -> pd.DataFrame   # obs_idx, arc_id
```

C also ships `src/evaluation/route_metrics.py` early (route continuity, pointwise accuracy, sequence metrics). **B needs `route_continuity()` for the comparison table**, so C delivers it in P1.

### 4.4 End-to-end pipeline and API (D)

```python
pipeline.run_trip(trip_dir) -> TripPrediction   # load -> candidates -> features -> score -> decode
```

The inference path must **never read `.route`**. Only `api/evaluation.py` reads reference data, to compute metrics server-side.

```
GET  /health
GET  /trips                    -> {"trips": ["<id>", ...]}                 # the ten reserved trips
POST /predict {"trip_id": ".."} -> {trip_id, n_obs, observations, predicted_arcs, predicted_geometry,
                                    metrics: {top1_acc, route_continuity, ...}, timing_ms}
```

The response must not contain the reference route or per-point labels.

## 5. Phase plan

Time-boxes are left for the team to fill, because the hackathon schedule is not in the briefing. **The briefing says there is a first-phase evaluation before lunch but not what it covers.** Ask the organisers, and until then treat G2 as the lunch deadline.

| Phase | Time-box | Goal | A | B | C | D | Gate |
|---|---|---|---|---|---|---|---|
| **P0** Kickoff | `__` | Shared understanding | Read all 5 PDFs together, then lock the decisions in section 9 | | | | everyone can state the problem and the contracts |
| **P1** Foundations | `__` | Thin slice end to end | Loaders, quality checks, local CRS, label alignment, candidate generation, features for 5 trips | Eval harness (GroupKFold, metrics, logging) | Metrics module, HMM skeleton with a distance-based stub scorer | Dependency files, API and UI skeletons against a stub scorer | **G1:** schema frozen, 5-trip sample table committed in `tests/fixtures/` |
| **P2** Experiments | `__` | Full data, 5+ models | Features for all 90 trips, EDA, then KNN and MLP | LR, RF, Boosting, comparison table and plots | HMM on real scores, decoder comparison | Scorer and decoder hooked into `pipeline.run_trip` | **G2 (phase-1 evaluation):** 5+ models with CV and student-test results, EDA done, one Folium map |
| **P3** Select and tune | `__` | Best model, tuned | Candidate radius/K sweep write-up | Pick the model, run Optuna on 80 trips, refit, tuned vs untuned | Tune decoder params on train + student test | API loads real artifacts, structured request logs | **G3:** tuned model exported, decoder params frozen |
| **P4** Integrate | `__` | Whole system runs | Fix feature or data bugs found downstream | Artifact fixes | Latency work on large trips | All 10 student-test trips pass through the API and UI, two-laptop dry run | **G4:** green run on both laptops |
| **P5** Report and freeze | `__` | PDF and repo done | EDA section | Model sections, tuning | HMM and Folium sections | README, API/UI screenshots, assemble PDF, secrets scan | **G5:** code freeze, every report number traces to `results/` |
| **P6** Rehearsal | `__` | Viva and demo | Q&A drill: every person explains every stage | | | Rehearse the evaluator flow with API logs on screen | ready |

**Critical path:** A's candidate table (G1) blocks B's real training and C's real scores. B's trained scorer blocks D's final integration. That is why P1 uses a stub scorer and a 5-trip sample, so B, C and D never sit idle.

## 6. Task lists per workstream

### A: Data, candidates, features, EDA
Code: `src/data_processing/`, `src/features/`, `notebooks/`, `src/visualization/eda_maps.py`

- [ ] Loaders for `.track`, `.nodes`, `.arcs`, `.route` (headerless, whitespace-separated, `lon lat t`). Check coordinate order.
- [ ] Dataset inventory CSV for train and student test: trip counts, file counts, points per trip, file sizes, dtypes.
- [ ] Quality report: missing or invalid points, timestamp order, irregular gaps, large-gap trips, geographic extent per trip.
- [ ] Road-network stats per trip: nodes, arcs, connectivity, isolated nodes, disconnected components, directedness.
- [ ] Per-trip local metric CRS, with the choice justified in the report.
- [ ] **Label alignment (Risk 1):** derive per-point reference arcs from the route. Record the method, its assumptions and failure counts.
- [ ] Candidate generation (spatial index, radius and/or top-K). Sweep radius/K against candidate miss rate on train data only.
- [ ] Feature engineering: point-to-arc distance, position along the arc, heading difference, speed, arc length, candidate rank, local connectivity, time gap. Fit any scaler on train only.
- [ ] Write the feature tables, `obs_labels.parquet`, `feature_schema.json` and `configs/folds.json`.
- [ ] EDA plots (missing values, gap distribution, trajectories, network size) and at least one Folium map of GPS plus network. State the partition used for each analysis.
- [ ] Train KNN and MLP through B's harness, then write their report subsections.

Done when: the feature tables regenerate from raw data with one command, and the inventory numbers match our own EDA.

### B: Models, evaluation, tuning, artifacts
Code: `src/models/`, `src/evaluation/`, `src/models/tuning/`. Outputs: `results/metrics/`, `results/plots/`

- [ ] Evaluation harness: GroupKFold CV by trip, precision/recall/F1/Macro-F1, log loss, observation-level Top-1, candidate miss rate, route continuity (from C), train and inference time, fixed seeds, logged configs.
- [ ] Train LR, Random Forest and Boosting, record the configs and logs, and write their 9-question report subsections. Add the spare model if time allows.
- [ ] Consolidated comparison table (report section 3.1) plus plots: CV Macro-F1 across models, student-test Top-1 and Macro-F1, log loss, confusion matrices. Loss curves only for models that produce epoch histories (MLP, boosting). Never fabricate a curve.
- [ ] Choose the primary selection metric, justify it, and pick the final model on CV and student-test evidence (never on training accuracy).
- [ ] Bayesian optimisation (Optuna TPE or similar) with group-aware CV on the 80 trips. Record the search space, objective, trial count, all trials, best params and compute cost.
- [ ] Refit, then compare tuned vs untuned on the same metrics and partitions (CV Macro-F1, student Top-1, student Macro-F1, log loss, inference time).
- [ ] Export `model.joblib`, `preprocessor.joblib`, `feature_schema.json` and `model_card.json` into `models/`. Check that probabilities are usable as HMM emissions, and calibrate if they are not.

Done when: `python -m src.models.run_all` reproduces the comparison table, and D can load the artifacts through the section 4.2 contract.

### C: HMM/Viterbi, route metrics, prediction maps
Code: `src/hmm/`, `src/evaluation/route_metrics.py`, `src/visualization/folium_maps.py`. Outputs: `maps/`, `results/metrics/`

- [ ] `src/evaluation/route_metrics.py` first: route continuity (consecutive predicted arcs identical or connected), pointwise accuracy, sequence-level metrics.
- [ ] HMM design, documented for the report. **Hidden states:** the candidate arcs of each point. **Emissions:** the scorer's probability. **Transitions:** road-network connectivity and route distance vs observed GPS displacement.
- [ ] Viterbi in log space with a clear recurrence. Cache bounded shortest-path lookups so large trips stay fast.
- [ ] Edge cases: empty candidate sets, large time gaps, consecutive observations on disconnected segments (penalise, bridge or restart, and state which).
- [ ] Decoder parameter tuning on train + student-test data only, with the grid and results recorded.
- [ ] Experiment: classifier-only vs classifier + HMM on pointwise accuracy, sequence metrics and continuity. Include cases where the HMM did not help.
- [ ] Folium prediction maps with layers for raw GPS, road network, reference route (offline report only), classifier-only route and HMM route. Add distinct colours, a legend and `LayerControl`.
- [ ] Commit the interactive HTML for at least one student-test trip under `maps/`.

Done when: `decode()` meets the section 4.3 contract and decodes the largest student-test trip (3,958 points) in an API-acceptable time. Measure it and record it.

### D: API, UI, deployment, integration, report
Code: `api/`, `ui/`, `src/pipeline.py`, `README.md`, `requirements*.txt`, `report/`

- [ ] Dependency files (`requirements.txt`), plus the new `src/visualization/` and `src/pipeline.py`, in P1. The scaffold and `.gitignore` already exist.
- [ ] FastAPI app with `/health`, `/trips`, `/predict` per section 4.4. Load artifacts once at startup. Read the data directory from config (`EVAL_DATA_DIR`) so student test stands in until the evaluator set arrives. Bind to `0.0.0.0`, enable CORS.
- [ ] Server-side evaluation in `api/evaluation.py`. Reference data never leaves the server and never reaches the model.
- [ ] Request logging that is readable on screen: trip id, point count, latency, accuracy. The briefing requires showing API logs on Laptop 1.
- [ ] UI for Laptop 2 with a configurable `API_URL`. It lists the ten trips, calls `/predict`, and shows GPS points, predicted route and metrics on a map. Handle loading states and timeouts. **Do not display the reference route.**
- [ ] Two-laptop network setup: same network, firewall, IP address, and a written fallback if the network fails.
- [ ] Integration test: all 10 student-test trips through the API and UI, with latencies recorded.
- [ ] README: install, run, deploy, repo layout.
- [ ] Screenshots of the working UI and of API responses in the terminal of the receiving laptop.
- [ ] Assemble the single consolidated PDF. Cross-check every number against `results/`. Run the secrets scan before pushing.

Done when: a clean checkout plus the README gets API and UI running on two machines, and all 10 trips return predictions.

## 7. Report ownership

Mapped to the required sections in `Deliverables.pdf`.

| Report section | Writes it | Needs from |
|---|---|---|
| 2.2 Dataset understanding and EDA (preprocessing, patterns, features, missing values, trajectory and network plots, Folium) | A | |
| 3.1 Consolidated model comparison table | B | A and C (their model rows, route continuity) |
| 3.2 Per-model subsections (9 questions each) and 3.3 visualisations | Whoever owns the model (A: KNN, MLP. B: LR, RF, Boosting) | |
| 4 Bayesian hyperparameter optimisation | B | |
| 5 HMM and Viterbi | C | B (scorer) |
| 6 Folium geographical visualisation | C (A supplies the EDA maps) | |
| 7 API, UI and deployment | D | |

Every graph and table must come from actual runs saved under `results/`. Answers must tie to our own experiments, not textbook descriptions.

## 8. Repository layout

The report requires clearly named folders, a README, and code that produces every submitted result. This is the scaffold already in the repo, with two additions marked NEW.

```
.github/
  HACKATHON_PLAN.md          this file
  statement/pathsnap/        the five briefing PDFs
data/
  raw/                       organiser's trips (train/, student_test/). LOCAL ONLY, gitignored
  processed/                 feature tables, regenerable. LOCAL ONLY, gitignored
src/
  data_processing/           A: loaders, quality checks, CRS
  features/                  A: candidates, features, label alignment
  models/                    B: one training script per model
    tuning/                  B: Bayesian optimisation (NEW subfolder)
  evaluation/                B: evaluation harness. C: route_metrics.py
  hmm/                       C: HMM and Viterbi
  visualization/             A: EDA maps and plots. C: Folium prediction maps (NEW)
  pipeline.py                D: run_trip(), the one inference entry point (NEW)
api/                         D: FastAPI app, server-side evaluation
ui/                          D: evaluator interface
models/                      exported weights, preprocessor, feature_schema.json, model_card.json
configs/                     folds.json, search spaces, HMM params
maps/                        interactive Folium HTML maps (committed)
results/
  metrics/                   CSV and JSON results, experiment and tuning logs (committed)
  plots/                     figures used in the report
notebooks/                   EDA and experiment notebooks
report/                      the technical report
tests/                       tests, plus small fixtures in tests/fixtures/
README.md                    D
```

Note the two similarly named folders: `src/models/` is training **code**, top-level `models/` holds the exported **artifacts**.

Git: one branch per workstream, merged into the default branch by PR. Commit often. The briefing says repository history is compared and the final code must match what we demo, so do not force-push over history.

## 9. Open decisions (settle in P0)

- [ ] Team size and names, so the roster and viva buddies are final.
- [ ] Hackathon schedule, and what the pre-lunch phase-1 evaluation covers.
- [ ] Primary model-selection metric and why. The report table needs CV Macro-F1, route-matching needs Top-1 and continuity, so pick one as primary and explain it.
- [ ] Definition of route continuity and of sequence-level performance (C proposes, everyone agrees).
- [ ] How the evaluator's ten trips reach Laptop 1 on the day (location, and whether the format matches `student_test`). Until confirmed, the API reads a configurable path and we test with student test.
- [x] Raw data is **not** committed. It lives locally in `data/raw/{train,student_test}/` (about 100 MB) and is gitignored, as are the derived tables in `data/processed/`. Each teammate copies the organiser's folder there. Document this in the README.
- [ ] UI framework (a simple Streamlit app with a Folium component is the quickest path).

## 10. Risks

| # | Risk | Owner | Mitigation |
|---|---|---|---|
| 1 | Per-point labels do not exist, only an arc route with fewer arcs than points, and some reference links are broken | A | Decide the alignment method in P1, document assumptions, report failure counts, avoid overstating label certainty |
| 2 | Trips are spread over several regions | A | Per-trip local CRS, never degrees |
| 3 | Large trips make candidate generation and Viterbi slow, hurting the live demo | A, C, D | Spatial index, cap K, cache shortest paths, measure the largest trips early |
| 4 | Candidate miss rate sets a ceiling on accuracy | A | Sweep radius/K, report it |
| 5 | Evaluator data format or location differs from what we assume | D | Configurable data path, confirm with organisers |
| 6 | B is overloaded | all | A takes KNN and MLP, and the spare model is optional |
| 7 | Data leakage or hardcoding flagged | all | Section 11 rules, code review before G5 |

## 11. Rules we must not break

- Evaluator-test data and labels are never used for training, feature selection, algorithm or hyperparameter choice, or decoder tuning. Use the 80 training trips and the 10 student-test trips only.
- No trip is split across partitions. CV is grouped by `trip_id`. Learned preprocessing is fitted on training data only.
- No hardcoded outputs. No bypassing the API. Reference labels stay on the server and never reach the model or UI.
- No secrets in the repo (passwords, API keys, tokens).
- Do not share models, weights, configs, code or results with other teams.
- Report only results we actually ran, and do not fabricate metrics, curves or citations. Every member must be able to explain every claim.
- Disclose AI assistance as the event policy requires. This plan was drafted with AI assistance (Claude Code) from the briefing PDFs.

## 12. Submission checklist

- [ ] Public GitHub repo containing data processing, features, training for every model, evaluation, Bayesian optimisation, HMM and Viterbi, API, UI, deployment config, Folium code, supporting CSVs and logs.
- [ ] Saved model weights and config files (within size limits).
- [ ] README with install, run and deploy instructions.
- [ ] One consolidated PDF report that follows `Deliverables.pdf`, including a Folium map screenshot, UI screenshots and API-response screenshots from the receiving laptop.
- [ ] All 10 student-test trips work through API and UI on two machines, with API logs visible.
- [ ] Secrets scan clean, and the demonstrated code matches the repo.
