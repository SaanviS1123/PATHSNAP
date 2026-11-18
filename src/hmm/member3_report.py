"""Measured Member 3 report source and comparison/latency figures."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.training import ROOT
from src.data_processing.member1_report import markdown_table


def generate_report():
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pathsnap-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=ROOT/'results/metrics';plots=ROOT/'results/plots'
    config=json.loads((ROOT/'configs/hmm.json').read_text())
    params=config['parameters']
    comparison=pd.read_csv(metrics/'hmm_comparison.csv')
    trips=pd.read_csv(metrics/'hmm_comparison_by_trip.csv',dtype={'trip_id':str})
    timing=pd.read_csv(metrics/'decode_timing.csv',dtype={'trip_id':str})
    tuning=pd.read_csv(metrics/'hmm_tuning.csv')
    model=json.loads((metrics/'hmm_tuning_model.json').read_text())
    student=comparison[comparison.partition=='student_test'].set_index('variant')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    order=['classifier','classifier_bridged','hmm']
    names=['Classifier','Classifier + bridges','HMM + bridges']
    for ax,column,title in zip(axes,['pointwise_accuracy','route_continuity','sequence_f1'],['Known-label point accuracy','Direct arc continuity','Ordered route LCS F1']):
        ax.bar(names,student.loc[order,column])
        ax.set(title=title,ylim=(0,1))
        ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(plots/'hmm_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for gap,group in tuning.groupby('gap_restart_s'):
        ax.plot(group.transition_beta_m,group.pointwise_accuracy,marker='o',label=f'restart > {gap:g} s')
    ax.set(title='Decoder tuning: training fold 3 only',xlabel='Transition beta (m)',ylabel='Known-label point accuracy')
    ax.legend();fig.tight_layout();fig.savefig(plots/'hmm_tuning.png',dpi=150);plt.close(fig)
    subset=timing[timing.partition=='student_test']
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(subset.trip_id,subset.total_ms/1000,label='Cold end-to-end')
    ax.bar(subset.trip_id,subset.warm_viterbi_ms/1000,label='Warm Viterbi')
    ax.set(title='Student-trip inference latency',ylabel='Seconds')
    ax.tick_params(axis='x',rotation=45);ax.legend()
    fig.tight_layout();fig.savefig(plots/'hmm_latency.png',dpi=150);plt.close(fig)
    pivot=trips[trips.partition=='student_test'].pivot(index='trip_id',columns='variant',values='pointwise_accuracy')
    pivot['hmm_change']=pivot.hmm-pivot.classifier
    pivot.reset_index().to_csv(metrics/'hmm_pointwise_changes.csv',index=False)
    worst=pivot.sort_values('hmm_change').reset_index().head(3)
    hmm=student.loc['hmm'];baseline=student.loc['classifier'];bridged=student.loc['classifier_bridged']
    largest=timing[timing.partition=='student_test'].sort_values('n_obs').iloc[-1]
    large_train=timing[timing.partition=='train_speed_only'].iloc[0]
    measured=trips[(trips.partition=='student_test')&(trips.variant=='hmm')].merge(
        trips[(trips.partition=='student_test')&(trips.variant=='classifier')][['trip_id','n_correct']],on='trip_id',suffixes=('','_classifier'))
    n_worse=int((measured.n_correct<measured.n_correct_classifier).sum())
    table_columns=['partition','variant','n_known_obs','n_unknown_obs','pointwise_accuracy','route_continuity','sequence_precision','sequence_recall','sequence_f1','decoder_restarts']
    text=f'''# Member 3 report source: HMM, Viterbi and route maps

Sections 5 and 6 for Member 4's single consolidated PDF. Code and prose were
prepared with Codex AI assistance. Every number below is from executed local
runs. The evaluator partition was not accessed.

## 5. HMM/Viterbi route reconstruction

### Hidden states, emissions and transition weights

A state is a directed candidate arc at one GPS observation. IDs remain zero-based
rows of that trip's `.arcs` file. Candidate generation and the classifier retain
the frozen Member 1 schema and Member 2 final model. There is no global arc-ID
classifier and no reference-route input during prediction.

The calibrated classifier's positive probabilities are clipped away from zero,
normalised within each observation and logged. Per-observation normalisation is
a common factor across all paths and therefore does not change the Viterbi
argmax. These are discriminative emission potentials, not a claim that the
classifier learned a generative GPS likelihood.

Directed route distance includes the remaining fraction of the previous arc,
a shortest directed path from its end node to the next arc's start node, and
the travelled fraction of the next arc. On the same arc, forward progress uses
its direct along-arc distance; backward motion up to
{params['backtrack_tolerance_m']:g} m is treated as GPS jitter. Larger reverse
movement needs a directed loop. Parallel directed arcs are retained, not summed
into an inflated graph-matrix edge weight.

The transition log weight is `-|route_distance - GPS_displacement| / beta` for
reachable pairs and negative infinity for impossible pairs. These exponential
weights are not row-normalised empirical traffic transition probabilities.
The recurrence is a weighted HMM-style chain:

`delta[t,j] = log_emission[t,j] + max_i(delta[t-1,i] + log_transition[t,i,j])`.

Backpointers recover the globally best candidate chain within each continuous
segment. A shared score offset is subtracted at each step to keep arithmetic
stable; it preserves the path argmax. Candidate ordering makes score ties
choose lower arc IDs deterministically. Tests compare the recurrence with
exhaustive enumeration on a small chain.

This design follows the road-distance versus observed-displacement principle in
[Newson and Krumm (2009)](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/),
using our learned emissions, directed endpoint handling and explicit restart
rules. We do not claim to reproduce their experiment or use their dataset.

### Missing candidates, gaps and disconnected roads

Actual feature generation supplies a nearest-arc fallback when the radius is
empty. The decoder additionally handles an empty observation by returning arc
`-1` and restarting afterwards. This sentinel is never used as a Python negative
network index. Actual student runs must still predict every observation.
Nonpositive time intervals, gaps above the selected threshold, spatial jumps
above {params['max_route_distance_m']:g} m and chains with no feasible transition
start a new segment. Restart reasons are recorded. Impossible transitions do not
receive an invented road connection or straight connector geometry.

Dijkstra searches are bounded by GPS displacement plus
{params['path_margin_m']:g} m, capped at {params['max_route_distance_m']:g} m.
Source searches use 100 m cutoff buckets and an LRU cache; reconstructed paths
are checked against the exact cutoff. Routes outside this bounded search are
considered unavailable, not proven globally disconnected. A 16-candidate cap
limits dynamic-programming state size; all selected-candidate tables retain IDs.

### Decoder tuning and data separation

A separate scorer used {len(model['base_fit_trip_ids'])} training trips for base
fitting, {len(model['calibration_trip_ids'])} disjoint trips for sigmoid
calibration, and {len(model['tuning_trip_ids'])} further training trips for decoder
tuning. All trip sets are recorded in `hmm_tuning_model.json`. This avoids tuning
on predictions from the final classifier fitted on all 80 trips. The base family
and parameters were selected earlier using training CV, so this is not independent
of all prior model selection and is not nested CV.

The grid used beta `[10,30,60,120]` m and restart gaps `[10,30,60]` s: twelve
configurations, with all observations from all sixteen tuning trips. Primary
selection was known-label micro point accuracy. Ties use direct continuity,
fewer restarts, larger beta, then a smaller restart gap. Student test did not
select parameters. Frozen settings are in `configs/hmm.json`:

```json
{json.dumps(params,indent=2)}
```

![Measured decoder tuning](../../results/plots/hmm_tuning.png)

### Classifier-only versus decoding

Point accuracy compares every known observation label, including candidate
misses; unknown aligned labels are excluded and counted. Those labels are
approximate route-to-point supervision, not independently verified physical
truth. Distance/heading also influenced alignment, so high agreement has that
limitation. Reference-route imperfections remain documented in Member 1's EDA.

Direct continuity keeps B's definition: identical arcs or exact directed
head-to-tail adjacency between consecutive observation predictions. All points,
including unknown labels and time gaps, participate. This is intentionally
stricter than being connected by an intermediate road path.

Ordered sequence precision/recall use the longest common subsequence of arc
occurrences against the original reference `.route`; F1 is their harmonic mean.
Repeated arc IDs remain distinct ordered occurrences. This is neither a set
Jaccard score nor edit accuracy. Flattening restarted segments for LCS preserves
arc order but does not itself penalise a break; restart/segment counts and direct
continuity are therefore reported separately. Sequence endpoints may differ when
GPS covers only part of the supplied route.

The bridge-only control applies identical bounded path reconstruction to greedy
classifier choices. It leaves point labels unchanged and reveals how much sequence
improvement comes from inserting intermediate road arcs rather than Viterbi.

{markdown_table(comparison[table_columns])}

On student test, classifier accuracy was {baseline.pointwise_accuracy:.4%}; HMM
accuracy was {hmm.pointwise_accuracy:.4%}. Direct continuity changed from
{baseline.route_continuity:.4%} to {hmm.route_continuity:.4%}. Ordered route F1
was {baseline.sequence_f1:.4f} for raw classifier arcs,
{bridged.sequence_f1:.4f} with bridging alone and {hmm.sequence_f1:.4f} with HMM
and bridging. Tuning-set results are development scores after parameter selection,
not held-out performance estimates.

![Measured classifier/bridge/HMM comparison](../../results/plots/hmm_comparison.png)

### Cases where HMM did not help

HMM had lower point accuracy than the classifier on {n_worse} student trips.
The lowest per-trip changes were:

{markdown_table(worst)}

These cases must be interpreted alongside the measured changes, not hidden by
an aggregate. Hard topology constraints can favour a connected alternative when
candidate coverage, the road network, scorer probabilities or aligned labels are
wrong. A bounded search can also miss a longer valid detour. Per-trip rows, route
F1 and explicit restarts are in `hmm_comparison_by_trip.csv`; they do not establish
which input is physically correct in an ambiguous case.

### Latency and large trips

The largest student trip `{largest.trip_id}` has {int(largest.n_obs):,} points and
{int(largest.n_arcs):,} directed arcs. Its cold end-to-end inference took
{largest.total_ms/1000:.3f} s, including loading/projection, feature generation,
scoring, transition preparation, Viterbi and route expansion. With transitions
already prepared, Viterbi took {largest.warm_viterbi_ms/1000:.3f} s. Model loading
is excluded when the batch reuses the startup scorer, as the future API should.
The provisional cold student target is 10 s, not an agreement invented with
Member 4. Maximum observed student cold time was {subset.total_ms.max()/1000:.3f} s.

The largest training trip `{large_train.trip_id}` has
{int(large_train.n_obs):,} points. Its cold total was
{large_train.total_ms/1000:.3f} s and warm Viterbi
{large_train.warm_viterbi_ms/1000:.3f} s. It is used for speed only; its classifier
was fitted on training data, so its quality is not presented as generalisation.
A separate warm-graph cap-8 diagnostic is recorded but does not change the frozen
16-state configuration or student results. Detailed stages/cache counts are in
`decode_timing.csv`.

![Measured student latency](../../results/plots/hmm_latency.png)

## 6. Geographical visualisation

All ten student maps are committed as `maps/pred_<trip_id>.html`. Layers include
raw GPS, cropped road geometry, classifier-only arc choices, HMM routes with
directed bridges, and the supplied reference route for the offline report only.
Colours, a legend, per-point tooltips, fitted bounds and layer controls distinguish
them. Predicted geometry consists of real road arcs; no straight road connector
is drawn across a restart. The generic map function adds a reference layer only
when explicitly supplied one by offline evaluation. API/UI inference must omit it.

Open [the largest student trip](../../maps/pred_{largest.trip_id}.html).
The rendered Folium screenshot, once generated by `src.visualization.render_maps`,
is `results/plots/hmm_folium_screenshot.png` for the consolidated report.

![Rendered Folium view](../../results/plots/hmm_folium_screenshot.png)

## Reproducibility and handoff

Run `python -m src.hmm.run_member3` after Members 1 and 2. Prediction uses
`src.hmm.predict.predict_trip()` or `decode(cands,network,params)` and never opens
`.route`. Offline comparison opens references only after prediction. The final
model is preserved; frozen decoder parameters, data/model/source fingerprints,
all grid rows, per-trip comparisons, timing logs and maps are retained. Member 4
still owns API/UI integration, two-laptop checks and final PDF assembly.
'''
    folder=ROOT/'report/member3';folder.mkdir(parents=True,exist_ok=True)
    (folder/'hmm_and_maps.md').write_text(text)
    print('Generated Member 3 report and figures',flush=True)


if __name__=='__main__':
    generate_report()
