# PathSnap: Team Assignments (4 members)

Who does what, in what order. This is the companion to [`HACKATHON_PLAN.md`](HACKATHON_PLAN.md), which holds the dataset facts, the interface contracts (section 4), the phases (P0 to P6) and the rules. Read that first. This file does not repeat it.

## 1. Roster

Fill in the names. Everything below refers to members by number.

| Member | Name | Workstream | Branch |
|---|---|---|---|
| **Member 1** | `<name>` | A: Data, candidates, features, EDA | `feat/data-features` |
| **Member 2** | `<name>` | B: Models, evaluation, Bayesian tuning, artifacts | `feat/models` |
| **Member 3** | `<name>` | C: HMM/Viterbi, route metrics, prediction maps | `feat/hmm` |
| **Member 4** | `<name>` | D: API, UI, deployment, integration, README, report | `feat/api-ui` |

## 2. At a glance

| | Member 1 (A) | Member 2 (B) | Member 3 (C) | Member 4 (D) |
|---|---|---|---|---|
| **Owns folders** | `src/data_processing/`, `src/features/`, `src/visualization/eda_maps.py`, `notebooks/` | `src/models/`, `src/evaluation/harness.py`, `src/models/tuning/`, `models/` | `src/hmm/`, `src/evaluation/route_metrics.py`, `src/visualization/folium_maps.py`, `maps/` | `api/`, `ui/`, `src/pipeline.py`, `report/`, `README.md`, `requirements.txt` |
| **Trains models** | KNN, MLP | Logistic Regression, Random Forest, Boosting | none | none (picks up a model only if someone falls behind, see section 9) |
| **Report sections** | 2.2 EDA, KNN and MLP subsections | 3.1 comparison table, LR/RF/Boosting subsections, 4 Bayesian optimisation | 5 HMM and Viterbi, 6 Folium | 7 API/UI/deployment, assembly of the whole PDF |
| **Blocks others with** | Feature tables (gate G1, then full set in P2) | Eval harness (P1), a first scorer (early P2), final model (G3) | `route_metrics.py` (P1), tuned decoder (G3) | Little. Others block D. |
| **Is blocked by** | Nobody | A's feature tables, C's `route_metrics.py` | A's feature tables, B's scorer | B's artifacts, C's decoder |

## 3. Rules so we do not step on each other

- **Edit only the folders you own.** A change to someone else's folder goes through them (or a PR they review).
- **Shared files** (`requirements.txt`, `README.md`, `configs/`, `.gitignore`): small PRs, merge fast, never reformat the whole file. Member 4 owns `requirements.txt` and `README.md`. Add your dependency or section by PR.
- **One branch per member** (names above), merged into `main` by PR. No force-pushes, because repository history is compared and the code we demo must match the repo.
- **Commit your own work under your own git identity.** Everyone must be able to explain their commits.
- **Tick your own boxes** in your section of this file, in your own PRs. Nobody edits another member's checklist, which avoids merge conflicts.
- **Interface changes** (feature schema, scorer, decoder, API response) are announced to the whole team before they are merged. See `HACKATHON_PLAN.md` section 4.
- **Results come from runs.** Every number in the report must trace to a file in `results/metrics/` produced by committed code.
- **Evaluator-test data is never touched** while developing. Use `data/raw/train/` and `data/raw/student_test/` only.

---

## 4. Member 1: Data, candidates, features, EDA (A)

You produce the table everyone else trains and decodes on, so your first priority is a thin working slice by gate **G1**, then the full set.

| ID | Task | Output | Phase / needs |
|---|---|---|---|
| A1 | Loaders for `.track`, `.nodes`, `.arcs`, `.route`. Headerless, whitespace-separated. Check coordinate order. | `src/data_processing/loaders.py` | P1 |
| A2 | Dataset inventory and quality report: trips per partition, points per trip, file sizes, dtypes, missing or invalid points, timestamp order and gaps, extent per trip, network size, connected components, isolated nodes, directedness | `results/metrics/inventory.csv`, `results/metrics/quality_report.csv`, `notebooks/01_eda.ipynb` | P1, needs A1 |
| A3 | Per-trip local metric CRS helper (trips span several regions, so no global CRS and no degrees). Test it on two trips from different regions. | `src/data_processing/geo.py` | P1, needs A1 |
| A4 | **Route to per-point labels.** A route has far fewer arcs than there are points, so design and document the alignment. Record failures and the unconnected reference links. | `src/features/labels.py`, `data/processed/obs_labels.parquet`, `results/metrics/label_alignment.csv` | P1, needs A1, A3 |
| A5 | Candidate generation with a spatial index. Sweep radius and K against candidate miss rate on train data only, then pick. | `src/features/candidates.py`, `results/metrics/candidate_sweep.csv` | P1, needs A3 |
| A6 | Feature builder plus the schema freeze: `feature_schema.json`, `configs/folds.json` (GroupKFold by `trip_id`, fixed seed), and a 5-trip fixture. **This is gate G1.** | `src/features/build.py`, `models/feature_schema.json`, `configs/folds.json`, `tests/fixtures/` | P1, needs A4, A5 |
| A7 | Full feature tables for all 90 trips, regenerated by one command | `data/processed/train.parquet`, `student_test.parquet` | P2, needs A6 |
| A8 | EDA plots (missing values, gaps, trajectories, network size) and at least one Folium map of GPS plus network. Say which partition each analysis used. | `results/plots/eda_*.png`, `maps/eda_*.html`, `src/visualization/eda_maps.py` | P2, needs A2 |
| A9 | Train **KNN** through B's harness (scaling matters for distance methods) | `src/models/train_knn.py` | P2, needs A7, B1 |
| A10 | Train **MLP** through B's harness. Loss curve only from the real epoch history. | `src/models/train_mlp.py` | P2 to P3, needs A7, B1 |
| A11 | Fix feature or label bugs reported by B and C. Write up the candidate-miss-rate ceiling. | commits plus a note in the report | P3 to P4 |
| A12 | Write report section 2.2 and the KNN and MLP subsections (the 9 questions each, plus visualisations) | `report/` | P5 |

**Done when:** `data/processed/*.parquet` regenerate from raw data with one command, the inventory matches your own EDA numbers, and the candidate miss rate is reported.

**You hand off:** feature schema and fixture (G1) to B, C and D. Full tables (P2) to B and C.

**Be ready to explain (viva):** how the raw files relate to each other, why a local CRS, how per-point labels were derived and their limits, what every feature means and why it was chosen, how candidates are chosen and what the miss rate costs.

---

## 5. Member 2: Models, evaluation, tuning, artifacts (B)

Your first deliverable is the evaluation harness, because Member 1 trains two models through it. Your second is a quick first scorer so Members 3 and 4 can integrate real probabilities early.

| ID | Task | Output | Phase / needs |
|---|---|---|---|
| B1 | **Evaluation harness.** GroupKFold using `configs/folds.json`; candidate-level precision, recall, F1, Macro-F1, log loss; observation-level Top-1; candidate miss rate; route continuity (from C1); train and inference time; seeds; config logging. Works with any scikit-learn style estimator. | `src/evaluation/harness.py`, `results/metrics/experiments.csv` | P1, needs the A6 fixture |
| B2 | Agree the **primary selection metric** with the team and write down why | note in `results/metrics/` or the report draft | P1 to P2 |
| B3 | **Logistic Regression** first (it is fast). Export a provisional `model.joblib` early so C and D can integrate real probabilities. | `src/models/train_logreg.py`, `models/` (provisional) | P2, needs A7, B1 |
| B4 | **Boosting** (HistGradientBoosting or LightGBM). Keep the eval history for loss curves. | `src/models/train_boosting.py` | P2 |
| B5 | **Random Forest** | `src/models/train_rf.py` | P2 |
| B6 | Comparison table and plots over all models, including Member 1's: CV Macro-F1, student-test Top-1 and Macro-F1, log loss, confusion matrices, loss curves only where real epoch histories exist | `results/metrics/model_comparison.csv`, `results/plots/` | P2, needs A9, A10 |
| B7 | Model selection with measured evidence (never training accuracy alone) | note plus table | P3 |
| B8 | **Bayesian optimisation** (Optuna TPE or similar) with group-aware CV on the 80 trips. Record space, objective, trial count, every trial, best params and compute cost. | `src/models/tuning/bayes_opt.py`, `results/metrics/tuning_trials.csv` | P3 |
| B9 | Refit and compare tuned vs untuned on the same metrics and partitions: CV Macro-F1, student Top-1, student Macro-F1, log loss, inference time | `results/metrics/tuned_vs_untuned.csv` | P3 |
| B10 | Export the final artifacts and check probability calibration, since the HMM uses the probabilities. **This is gate G3 for the model.** | `models/model.joblib`, `preprocessor.joblib`, `model_card.json` | P3 |
| B11 | Write report section 3.1, the LR, RF and Boosting subsections, and section 4 | `report/` | P5 |

**Done when:** one command reproduces the comparison table, and Member 4 can load `models/` through the scorer contract in `HACKATHON_PLAN.md` section 4.2.

**You hand off:** the harness (P1) to Member 1. A provisional scorer (early P2) and the final artifacts (G3) to Members 3 and 4.

**Be ready to explain (viva):** why each algorithm suits the problem, how the five-plus models compare, why the final model was chosen on that metric, what was searched in tuning and whether tuning helped, what the logs show about over- or underfitting.

---

## 6. Member 3: HMM, Viterbi, route metrics, prediction maps (C)

You can start with no model at all by using a distance-based stub scorer. Your first deliverable is `route_metrics.py`, because Member 2's comparison table needs route continuity.

| ID | Task | Output | Phase / needs |
|---|---|---|---|
| C1 | **Route metrics**: route continuity, pointwise accuracy, sequence-level metrics. Agree the definitions with the team in P0 and unit-test on synthetic arc sequences. | `src/evaluation/route_metrics.py` | P1 |
| C2 | Road network class: directed arc graph from `.arcs` and `.nodes`, successor lookup, bounded shortest-path distance between arcs, cached | `src/hmm/network.py` | P1 |
| C3 | Viterbi skeleton in log space with emission and transition modules, run end to end on the fixture with a stub scorer | `src/hmm/viterbi.py` | P1, needs C2, the A6 fixture |
| C4 | Decode with real scores. Handle empty candidate sets, large time gaps and consecutive observations on disconnected segments. Write down which of penalise, bridge or restart you chose. | `src/hmm/` | P2, needs B3, A7 |
| C5 | Tune decoder parameters on train plus student-test data only. Save the grid and results. | `results/metrics/hmm_tuning.csv`, `configs/hmm.json` | P2 to P3 |
| C6 | **Classifier-only vs classifier + HMM** on pointwise accuracy, sequence metrics and continuity. Include the cases where the HMM did not help. | `results/metrics/hmm_comparison.csv` | P3, needs the final model |
| C7 | **Speed on large trips.** The largest student-test trip has 3,958 points and the largest train trip 19,313. Measure, then optimise (candidate cap, cached path lookups). | `results/metrics/decode_timing.csv` | P3 to P4 |
| C8 | Folium prediction maps with layers for raw GPS, road network, reference route (offline report only), classifier-only route and HMM route. Distinct colours, legend, `LayerControl`. Commit the HTML for at least one student-test trip, ideally all ten. | `src/visualization/folium_maps.py`, `maps/pred_*.html` | P2 to P3 |
| C9 | Write report sections 5 and 6 | `report/` | P5 |

**Done when:** `decode()` meets the contract in `HACKATHON_PLAN.md` section 4.3, and the largest student-test trip decodes in a time the API can live with (measure it, then agree a target with Member 4).

**You hand off:** `route_metrics.py` (P1) to Member 2. The decoder and frozen params (G3) to Member 4. Maps and comparison numbers to the report.

**Be ready to explain (viva):** the hidden states, how emissions and transitions are computed, the Viterbi recurrence, how gaps and disconnected segments are handled, where the HMM helped and where it did not.

---

## 7. Member 4: API, UI, deployment, integration, report (D)

You build against stubs first, so you are never waiting. You also own the thing the evaluator actually sees, and the final PDF.

| ID | Task | Output | Phase / needs |
|---|---|---|---|
| D1 | Dependency files and environment notes: pinned `requirements.txt`, Python version, `.env.example` with no real secrets | `requirements.txt`, `.env.example` | P1 |
| D2 | API skeleton: `/health`, `/trips`, `/predict`, config for `EVAL_DATA_DIR` and `MODEL_DIR`, CORS, bind `0.0.0.0`. Per-request log line (trip id, point count, latency, accuracy) readable on screen. | `api/main.py`, `api/schemas.py`, `api/config.py` | P1 |
| D3 | UI skeleton: lists the trips from `/trips`, calls `/predict`, shows GPS points plus predicted route plus metrics on a map, `API_URL` configurable, loading and timeout handling. **It must not display the reference route.** | `ui/app.py` | P1 |
| D4 | `run_trip()`: load, candidates, features, score, decode. First with stubs, then real parts. The inference path never opens `.route`, and a test enforces it. | `src/pipeline.py`, `tests/` | P1 to P2, needs A6, C3 |
| D5 | Server-side evaluation: reads the reference route, computes metrics with C's functions, never puts reference data in a response. A test checks the response schema has no reference fields. | `api/evaluation.py`, `tests/` | P2, needs C1 |
| D6 | Wire in the real artifacts: load `models/` and the HMM params once at startup, check the feature schema version matches | `api/`, `src/pipeline.py` | P3, needs B10, C5 |
| D7 | **Two-laptop setup**: same network or hotspot, firewall open, `curl` from the second laptop, written fallback if the network fails | notes in `README.md` | P4 |
| D8 | **Integration run**: all ten student-test trips through the API and UI. Record latencies. Take screenshots of the UI and of the API log on the receiving laptop's terminal. | `results/metrics/integration_run.csv`, `report/figures/` | P4 |
| D9 | README (install, run, deploy, where to put the data), repo hygiene, secrets scan, check no committed file is near GitHub's 100 MB limit | `README.md` | P5 |
| D10 | Write report section 7, then **assemble the whole PDF** and audit that every number traces to `results/metrics/` | `report/` | P5 |

**Done when:** a clean clone plus the README gets the API and UI running on two machines, and all ten trips return predictions with the API log visible.

**You hand off:** nothing blocks others, but the final integration depends on B10 and C5, so check on them at each gate.

**Be ready to explain (viva):** the end-to-end request flow, why labels never reach the model or UI, how the API is deployed across two laptops, what the logs show for a request.

---

## 8. Handoffs and critical path

| # | Handoff | From → To | Artifact | Needed by |
|---|---|---|---|---|
| 1 | Feature schema plus 5-trip fixture (**G1**) | 1 → 2, 3, 4 | `models/feature_schema.json`, `tests/fixtures/` | end of P1 |
| 2 | Route metrics | 3 → 2 | `src/evaluation/route_metrics.py` | P1, for the comparison table |
| 3 | Evaluation harness | 2 → 1 | `src/evaluation/harness.py` | P1, for KNN and MLP |
| 4 | Full feature tables | 1 → 2, 3 | `data/processed/*.parquet` | start of P2 |
| 5 | Provisional scorer (Logistic Regression) | 2 → 3, 4 | `models/model.joblib` | early P2 |
| 6 | Final model and artifacts (**G3**) | 2 → 4 | `models/` | end of P3 |
| 7 | Decoder and frozen params (**G3**) | 3 → 4 | `src/hmm/`, `configs/hmm.json` | end of P3 |
| 8 | Report sections | 1, 2, 3 → 4 | `report/` | P5 |

The longest chain is Member 1's feature tables, then Member 2's models, then Member 3's decoding, then Member 4's integration. Stubs and the 5-trip fixture exist so that nobody waits on the chain in P1. **Keep Member 1's G1 short and rough rather than late and perfect.**

## 9. If someone falls behind

Check at G2 (the phase-1 evaluation checkpoint). The valves, in order:

1. **Member 2 behind:** Random Forest moves to Member 4, who is the lightest in P1 and P2.
2. **Member 1 behind on models:** MLP is dropped for a cheaper spare (Gaussian NB or a calibrated linear SVM) and moves to Member 4. We still need five models in the report, so do not let the count fall below five.
3. **Member 3 blocked on scores:** keep using the stub scorer, and switch to the provisional Logistic Regression scorer as soon as it exists.
4. **Member 4 finishes early:** pair with Member 3 on decoder speed, write tests, and generate Folium maps for all ten student-test trips.

When a task moves, update this file in a PR and tell the team. The new owner also writes that report subsection and must be able to explain it.

## 10. Shared tasks (everyone)

| Phase | Task |
|---|---|
| **P0** | Everyone reads all five briefing PDFs. Settle the open decisions in `HACKATHON_PLAN.md` section 9. Agree the route-continuity and primary-metric definitions. |
| **Each gate** | 15-minute sync: what is done, what is blocked, what changed in an interface. |
| **P5** | Each member presents their stage to the others for 10 minutes. This is the cross-teach that the viva needs. |
| **P6** | Mock viva: each person is asked questions from the other members' lists above. Rehearse the evaluator flow with the API log on screen. |
| **All phases** | Disclose AI assistance as the event policy requires. Never submit analysis of an experiment we did not run. |

## 11. Final check

- [ ] Every member has committed their own work to the repo under their own identity.
- [ ] Every member can explain every stage at overview level, and their own stage in depth.
- [ ] G1 to G5 in `HACKATHON_PLAN.md` section 5 are met.
- [ ] The submission checklist in `HACKATHON_PLAN.md` section 12 is ticked.
