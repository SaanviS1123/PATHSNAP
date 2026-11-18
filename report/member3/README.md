# Member 3 handoff

Member 3 provides the directed graph, HMM/Viterbi, C1 route metrics, decoder
parameter selection, prediction maps and report source. Work is on `main`.
Decisions and measured outcomes are recorded in `DECISIONS.md`.

## Reproduce

Use Python 3.13 from the repository root after Member 1 data preparation and
Member 2's final model export. No evaluator data is used.

```bash
python -m pip install -r requirements-member3.txt
python -m src.hmm.run_member3
python -m pytest -q
```

The command trains a separate 48-trip scorer with 16 calibration trips and tunes
the decoder on the remaining 16 training trips. Model family/parameters came from
Member 2; this split is independent of the new scorer fit/calibration, not all
prior model selection. Twelve beta/gap configurations are evaluated using all
known labels in that tuning population. `configs/hmm.json` is frozen before the
final scorer predicts any student-test trip. Matching tuning scores/transition
caches are reused; source, data, schema, folds and model fingerprints guard them.

The runner evaluates all ten student trips, generates ten interactive maps,
compares raw classifier arcs, classifier plus bridging, and HMM plus bridging,
and measures cold inference/warm Viterbi on the largest student and training
trips. The 19,313-point training trip is used only as a speed case. Private caches
are under `data/processed/member3/`, not an inference dependency.

## Inference contract for Member 4

```python
from src.hmm.predict import load_params, predict_trip
from src.models.training import load_scorer

# Load these once at API startup.
scorer = load_scorer('models')
params = load_params('configs/hmm.json')
result = predict_trip('data/raw/student_test/00000056', scorer, params)

# One row for every GPS observation; actual student results have no -1 arcs.
per_point = result.decoded                 # obs_idx, arc_id
route_parts = result.segments              # directed arc sequences, split at breaks
geometry = result.network.metric.trip.nodes[result.network.arcs[route_parts[0]]]
# Do not draw a straight line between separate route_parts.
```

`predict_trip` loads `.track/.nodes/.arcs`, creates metric features, scores them,
decodes, and reconstructs directed road paths. It never opens `.route` or reads
processed reference labels. The returned network's `trip.route` is `None`.
`result.timing_ms` records loading/projection, features, scoring, transition
preparation, Viterbi, expansion and total. Startup model loading is excluded if
the caller supplies a preloaded scorer. The provisional cold student target is
10 s; it is not a claim that another member already agreed to a latency budget.

The lower-level plan contract is also available:

```python
from src.hmm.network import RoadNetwork
from src.hmm.viterbi import decode, HMMParams

network = RoadNetwork.from_trip_dir(trip_dir)
prediction = decode(candidates, network, params)
# candidates: trip_id, obs_idx, arc_id, score (no reference fields)
```

States are ordered by arc ID for deterministic ties. Emissions use logged,
per-observation-normalised classifier probabilities. Transitions penalise the
absolute difference between directed route distance and GPS displacement.
Partial-arc positions are included. At most 20 m backward progress on the same
arc is allowed as GPS jitter; larger reversal needs a directed loop. Searches use
a 500 m margin, a 5 km cap and bounded-source LRU caching. Search bounds imply
unavailable paths are not proof of global disconnection.

Empty candidates yield arc `-1`; large/nonpositive gaps, spatial jumps, empty
observations and impossible chains restart segments. Actual candidate generation
has nearest-arc fallback, and all student observations receive valid arc IDs.
`prediction.attrs['diagnostics']` lists restart reasons and counts. Reference
columns are rejected by the decoder. Offline evaluation reads labels/routes only
after prediction. No hardcoded predictions, reference geometry, or evaluator
labels enter the inference path.

## Metrics and evidence

`src/evaluation/route_metrics.py` exposes `route_continuity`,
`pointwise_accuracy`, `lcs_length`, and `sequence_metrics`. Direct continuity
adopts Member 2's definition, preserving prior results. It includes all points,
not just known reference labels. LCS F1 compares ordered arc occurrences;
repeated IDs are not collapsed into a set. It is not edit accuracy and does not
penalise restarts independently; restart/segment counts and continuity accompany
it. Approximate aligned point labels and imperfect reference routes still limit
physical interpretation.

Results: `hmm_tuning*.csv`, `hmm_tuning_model.json`, `hmm_comparison*.csv`,
`hmm_error_changes.csv`, `decode_timing.csv`, `hmm_cap_diagnostic.json` and
`member3_run.json`. The bridge-only control isolates path expansion from Viterbi.
The report includes all trip changes, including the trip that worsened. Student
cache counters include offline comparison work; latency stages cover inference
only. Member 2's model artifacts and original experiment records are preserved.

## Maps and screenshots

`maps/pred_<trip_id>.html` has GPS, road network, classifier, HMM and offline
reference layers. `prediction_map(result, path)` defaults to **no reference**;
only offline evaluation explicitly supplies one. Member 4 must omit reference
layers in the evaluator UI. Roads are actual network arcs; separate route
segments do not get an invented connector.

Restyle saved student predictions without rerunning model selection:

```bash
python -m src.visualization.folium_maps --offline-reference
```

For actual browser screenshots and verification (optional browser dependency,
not needed by the API):

```bash
python -m pip install -r requirements-member3-render.txt
python -m playwright install chromium --only-shell
python -m src.visualization.render_maps
```

If using a custom browser cache, set `PLAYWRIGHT_BROWSERS_PATH` for both install
and rendering, or pass `--browser-cache /path/to/cache` to the renderer. The local
run used `/private/tmp/pathsnap-browsers`. Public CDN/basemap access is needed.
The browser uses an isolated headless context, not a personal browser profile.
`folium_render_checks.json` records page errors, tile counts, browser version and
screenshot focus. Screenshots are actual rendered Folium views. The failure
view is selected from known classifier-correct/HMM-wrong observations after
parameters are frozen; it is an illustration, not parameter tuning.

`report/member3/hmm_and_maps.md` is the generated source for sections 5–6 of
Member 4's consolidated PDF. API/UI deployment, two-laptop checks and final PDF
assembly remain Member 4's work. Code and report drafting disclose Codex AI
assistance; all reported metrics come from executed local code.
