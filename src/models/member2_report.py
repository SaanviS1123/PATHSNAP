"""Member 2 report sections and figures generated exclusively from saved runs."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.training import ROOT,write_json
from src.data_processing.member1_report import markdown_table

NAMES={'logistic_regression':'Logistic regression','random_forest':'Random forest',
       'hist_gradient_boosting':'Hist boosting','knn':'KNN','mlp':'MLP'}


def display_name(name):
    if name.endswith('_tuned'):
        return NAMES.get(name[:-6],name[:-6])+' tuned'
    return NAMES.get(name,name)


def generate_report(records,selection,tuning):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pathsnap-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=ROOT/'results/metrics'
    plots=ROOT/'results/plots'
    plots.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(metrics/'model_comparison.csv')
    labels=[display_name(n) for n in table.model_name]
    for column,title,filename in [('cv_macro_f1','Five-fold candidate Macro-F1','model_cv_macro_f1.png'),
                                  ('cv_top1_accuracy','Five-fold observation Top-1','model_cv_top1.png'),
                                  ('student_log_loss','Student-test candidate log loss','model_log_loss.png'),
                                  ('student_route_continuity','Student-test direct classifier continuity','model_route_continuity.png')]:
        fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(labels,table[column])
        ax.set(title=title,ylabel=column)
        if 'loss' not in column:
            ax.set_ylim(0,1)
        ax.tick_params(axis='x',rotation=20)
        fig.tight_layout()
        fig.savefig(plots/filename,dpi=150)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    x=np.arange(len(table))
    ax.bar(x-.18,table.student_top1_accuracy,width=.36,label='Observation Top-1')
    ax.bar(x+.18,table.student_macro_f1,width=.36,label='Candidate Macro-F1')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1),title='Student-test comparison (known labels)')
    ax.tick_params(axis='x',rotation=20)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2)
    fig.tight_layout()
    fig.savefig(plots/'model_student_metrics.png',dpi=150)
    plt.close(fig)
    histories=[]
    for record in records:
        name=record['name']
        matrix=np.asarray(record['student']['confusion_matrix'])
        fig,ax=plt.subplots(figsize=(4,3.5))
        ax.imshow(matrix,cmap='Blues')
        for (i,j),value in np.ndenumerate(matrix):
            ax.text(j,i,f'{value:,}',ha='center',va='center')
        ax.set(title=f'{display_name(name)}: student test',xlabel='Predicted',ylabel='Reference',xticks=[0,1],yticks=[0,1])
        fig.tight_layout()
        fig.savefig(plots/f'confusion_{name}_student.png',dpi=150)
        plt.close(fig)
        for history in record['cv'].get('histories',[]):
            histories.extend(dict(model=name,run=f'fold_{history["fold"]}',kind=history['kind'],iteration=i+1,value=float(value)) for i,value in enumerate(history['values']))
    member1_history=metrics/'mlp_loss_history.csv'
    if member1_history.exists():
        for row in pd.read_csv(member1_history).itertuples():
            histories.append(dict(model='mlp',run=row.run,kind='training_objective',iteration=int(row.epoch),value=float(row.train_loss)))
    history_frame=pd.DataFrame(histories)
    history_frame.to_csv(metrics/'model_training_history.csv',index=False)
    if len(history_frame):
        for (name,kind),group in history_frame.groupby(['model','kind']):
            fig,ax=plt.subplots(figsize=(7,4))
            for run,history in group.groupby('run'):
                ax.plot(history.iteration,history.value,label=run)
            ax.set(title=f'{display_name(name)}: {kind}',xlabel='Iteration',ylabel=kind)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots/f'training_history_{name}.png',dpi=150)
            plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot([0,1],[0,1],linestyle='--',color='grey',label='Ideal')
    for record in records:
        if record['name'].endswith('_tuned'):
            continue
        bins=record['student']['calibration_bins']
        ax.plot([r['mean_probability'] for r in bins],[r['positive_fraction'] for r in bins],marker='o',markersize=3,label=display_name(record['name']))
    final_bins=selection['final_student_test']['calibration_bins']
    ax.plot([r['mean_probability'] for r in final_bins],[r['positive_fraction'] for r in final_bins],marker='s',color='black',label='Final export')
    ax.set(title='Student-test probability reliability (after selection)',xlabel='Mean positive probability',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots/'model_calibration.png',dpi=150)
    plt.close(fig)
    columns=['model_name','cv_macro_f1','cv_top1_accuracy','student_top1_accuracy','student_route_continuity','student_macro_f1','student_log_loss','student_inference_time_sec']
    compact=table[columns].copy()
    compact.columns=['Model','CV Macro-F1','CV Top-1','Student Top-1','Direct continuity','Student Macro-F1','Log loss','Inference (s)']
    audit=json.loads((metrics/'calibration_audit.json').read_text())
    final=selection['final_student_test']
    source=f'''# Member 2 report source: model comparison, experiments and tuning

These sections extend Member 1's report source for the single final team PDF.
All values are from local executed runs. Code and report drafting used Codex AI
assistance; the team remains responsible for understanding and verifying them.
No evaluator-test partition was read.

## 3.1 Consolidated comparison and selection

The five distinct families are Logistic Regression, KNN, Random Forest,
histogram gradient boosting and MLP. A tuned variant is a separate row, not a
sixth distinct baseline family. All use the same schema and frozen five trip
folds over the 80 training trips. Known-label training rows total 2,134,559;
20,540 training and 17 student-test observations have unknown alignment labels.
`-1` never enters supervised fitting or candidate metrics. Top-1 counts missing
reference candidates as incorrect and excludes only unknown reference labels.

{markdown_table(compact)}

The primary metric is mean fold observation Top-1: one candidate road arc must be
chosen per GPS observation. Candidate Macro-F1 measures performance despite many
negative rows; log loss, Brier score and reliability diagrams assess probability
quality for the decoder. Exact score ties choose the lower arc ID. Preprocessing
is fitted on known-label rows within each training fold only.

The measured baseline CV winner was **{selection['baseline_winner']}**.
The final family/configuration is **{selection['selected_model']}**.
Only CV scores choose the family and tuned-versus-untuned configuration; student
test is a reported development holdout, not an optimisation objective. Family
and search selection reuse CV, so these CV values are selection-optimistic, not
nested-CV estimates. Candidate radius/K was already chosen using training-wide
labels; CV remains conditional on that fixed candidate configuration.

Member 1 KNN/MLP results were reused after checking data, schema, folds and source
fingerprints. Their historical CV probabilities were not retained, so their CV
Brier/ECE/continuity remain unavailable. Their saved full-training models were
run on student test to measure those diagnostics consistently for all models.
No absent metric was fabricated or inferred from a confusion matrix.

Route continuity here is a classifier-only B diagnostic: adjacent observation
predictions use the same directed arc or connect head-to-tail. Unknown labels and
large gaps remain in this topology-only metric. Missing observation indexes break
continuity. Short intermediate road arcs can be skipped between GPS observations,
so this direct metric does not establish physical route infeasibility. C1's final
metric, HMM transitions and sequence evaluation remain Member 3's work.

![CV Macro-F1](../../results/plots/model_cv_macro_f1.png)
![CV Top-1](../../results/plots/model_cv_top1.png)
![Student metrics](../../results/plots/model_student_metrics.png)
![Log loss](../../results/plots/model_log_loss.png)
![Classifier continuity](../../results/plots/model_route_continuity.png)

## 3.2 Member 2 model subsections

Member 1's KNN and MLP nine-question sections remain in
`report/member1/eda_and_models.md`. Below are the three additional families.
'''
    explanations={
      'logistic_regression':('A scaled linear scorer gives a fast, interpretable reference for ranking nearby arcs.','It trains and scores compactly but cannot represent arbitrary nonlinear geometry/heading interactions. Balanced class weights affect its probability prior; calibration is assessed explicitly.','Weighted binary logistic loss with L2 regularisation, optimised by L-BFGS.'),
      'random_forest':('Trees can learn nonlinear interactions between distance, heading, rank and connectivity without scaling.','Bootstrap averaging reduces individual-tree variance. Depth, leaf size, tree count and bootstrap fraction bound compute and export size; probabilities can still be distorted by class weighting.','Bootstrap decision trees minimise weighted Gini impurity; their positive probabilities are averaged. It has no epoch loss curve.'),
      'hist_gradient_boosting':('Histogram boosting can learn nonlinear feature combinations with efficient tabular-data fitting.','Sequential tree corrections are expressive and compact. Learning rate, leaf count and iteration count control capacity. Row-level early stopping is disabled to keep trips out of internal mixed partitions.','Weighted binary log loss with L2 leaf regularisation; histogram-based gradient boosting.')}
    for record in records:
        name=record['name']
        if name not in explanations:
            continue
        cv,test=record['cv'],record['student']
        suitability,limitations,objective=explanations[name]
        diagnostic=record['training_diagnostic_macro_f1']
        gap=diagnostic-cv['macro_f1']
        source+=f'''
### {display_name(name)}

1. **Suitability.** {suitability}
2. **Strengths and limitations.** {limitations}
3. **Inputs.** The twelve frozen geometry/motion/connectivity features. Trip IDs, observation indexes, arc IDs, reference routes and labels are excluded from learned inputs.
4. **Preprocessing and validation.** Logistic Regression uses a fold-fitted StandardScaler; tree families use raw features. Five trip-grouped folds hold out 16 trips each; no row-level split mixes trips. Final refit used {test['n_fit_rows']:,} known candidate rows from {test['n_fit_trips']} trips.
5. **Initial configuration.** Exact recorded parameters follow below. The Random Forest's bootstrap fraction limits rows per tree, not the set of training trips admitted to the fit.
6. **Objective.** {objective}
7. **Student-test results.** Candidate accuracy {test['candidate_accuracy']:.4%}, positive precision {test['precision']:.4f}, recall {test['recall']:.4f}, F1 {test['f1']:.4f}; known observations {test['n_known_obs']:,}, unknown observations {test['n_unknown_obs']:,}; candidate miss rate {test['candidate_miss_rate']:.4%}.
8. **Required metrics.** Student Top-1 {test['top1_accuracy']:.4%}, Macro-F1 {test['macro_f1']:.4f}, log loss {test['log_loss']:.4f}. CV Macro-F1 {cv['macro_f1']:.4f} (fold SD {cv['macro_f1_std']:.4f}); CV Top-1 {cv['top1_accuracy']:.4%}. Refit {test['train_time_sec']:.3f} s; student inference {test['inference_time_sec']:.3f} s for {test['n_candidate_rows']:,} candidate rows. Timings exclude loading, feature generation and HMM decoding.
9. **Fit/generalisation evidence.** The <=10k fitted-row diagnostic Macro-F1 was {diagnostic:.4f}, a {gap:+.4f} difference from grouped CV. It is optimistic because these rows entered fitting. Fold Macro-F1 ranged {min(r['macro_f1'] for r in cv['folds']):.4f}–{max(r['macro_f1'] for r in cv['folds']):.4f}. This measured gap and fold spread guide overfitting assessment; training score alone cannot establish generalisation or rule out underfitting. Refit warnings: {test['warnings'] or 'none'}.

```json
{json.dumps(record['configuration'],indent=2,default=str)}
```

![{display_name(name)} student confusion matrix](../../results/plots/confusion_{name}_student.png)
'''
        if record['cv'].get('histories'):
            source+=f'\n![Recorded {name} iteration history](../../results/plots/training_history_{name}.png)\n'
    source+=f'''
## 3.3 Recorded histories and probability diagnostics

MLP curves come from Member 1's actual epoch objectives. Boosting curves are
actual per-stage empirical log losses on a fixed <=10k subset of that fold's
fitted rows; they are labelled `fit_sample_log_loss` and exclude the L2 penalty.
They are not validation-loss curves. Linear regression classifiers and forests
have no fabricated epoch histories. Every curve row is in
`results/metrics/model_training_history.csv`.

![Student probability reliability](../../results/plots/model_calibration.png)

## 4. Bayesian hyperparameter optimisation

Tuning targeted **{tuning['model_name']}**, the measured baseline CV winner.
Optuna TPE used seed {tuning['seed']} with {tuning['n_trials']} completed trials:
{tuning['n_startup_trials']} startup trials and {tuning['n_tpe_trials']} subsequent
TPE proposals. Every completed objective evaluated all five frozen folds over
known training labels. No student-test result entered the objective.

Search distributions:

```json
{json.dumps(tuning['search_space'],indent=2)}
```

Best trial: {tuning['best_trial']}, objective Top-1
{tuning['best_top1_accuracy']:.4%}. Search compute was
{tuning['total_compute_seconds']:.1f} s (sum of measured objective runtimes).
The SQLite study and sampler state stay in the regenerable processed-data cache;
all trial parameters, objectives, timings and states are in
`results/metrics/tuning_trials.csv`, with per-trial fold metrics under
`results/metrics/member2/trial_*.json`.

Best parameters:

```json
{json.dumps(tuning['best_params'],indent=2)}
```

Tuned versus untuned, using identical metric definitions and partitions:

{markdown_table(pd.read_csv(metrics/'tuned_vs_untuned.csv')[['model_name','cv_macro_f1','cv_top1_accuracy','student_top1_accuracy','student_macro_f1','student_log_loss','student_inference_time_sec']])}

The tuned configuration {'replaced' if selection['tuned_selected'] else 'did not replace'}
the untuned baseline. The rule requires strictly better CV Top-1, regardless of
which row has a better student-test result. Tuned CV is optimistic after search;
student test provides the separate reported development check.

### Calibration and final export

A training-only audit fitted the selected base estimator on
{len(audit['base_fit_trip_ids'])} trips, fitted a sigmoid on
{len(audit['calibration_trip_ids'])} disjoint trips, and evaluated probability
quality on {len(audit['audit_trip_ids'])} further disjoint trips. Its acceptance
rule requires a positive sigmoid slope, lower audit log loss and no higher audit
Brier score. This audit follows model selection and is not an unbiased estimate
of the entire selected pipeline. Student test never chooses calibration.

Audit raw/calibrated log loss: {audit['raw']['log_loss']:.6f} /
{audit['calibrated']['log_loss']:.6f}; Brier score:
{audit['raw']['brier_score']:.6f} / {audit['calibrated']['brier_score']:.6f}.
Calibration was **{'enabled' if selection['calibration_enabled'] else 'disabled'}**.
When enabled, the final sigmoid is fitted on known-label out-of-fold predictions
from all 80 training trips, then attached to the model refitted on all training
rows. Different base-fit sizes can shift probability distributions; held-out
student reliability is therefore reported, not assumed perfect.

Final exported student metrics: Top-1 {final['top1_accuracy']:.4%}, Macro-F1
{final['macro_f1']:.4f}, log loss {final['log_loss']:.6f}, Brier
{final['brier_score']:.6f}, ECE {final['ece_10']:.6f}, direct continuity
{final['route_continuity']:.4%}. Sigmoid calibration is increasing and preserves
candidate score ordering apart from floating-point ties; it changes threshold
metrics and probability quality, not the intended route-ranking objective.

`models/model.joblib` accepts raw DataFrame features and includes fitted
preprocessing and optional calibration. `preprocessor.joblib` is separately saved
for inspection/reconstruction; do not apply it before calling the exported scorer.
The schema and card include seed, inputs, configuration/selection and artifact
checksums. Model outputs are finite probabilities for log-space decoding.
The inference scorer reads no reference files. See the handoff README for usage.

References checked 18 November 2026:
[HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html),
[Optuna TPESampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
'''
    folder=ROOT/'report/member2'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'models_and_tuning.md').write_text(source)
    (metrics/'model_selection.md').write_text(f"# Final model selection\n\nSelected: **{selection['selected_model']}**.\n\nPrimary metric: frozen-fold observation Top-1. Student test did not select parameters.\n\nCalibration enabled: {selection['calibration_enabled']}.\n\nFinal metrics are in `model_selection.json`; full analysis is in `report/member2/models_and_tuning.md`.\n")
    decision=ROOT/'DECISIONS.md'
    decision_id='| D020 |'
    if decision_id not in decision.read_text():
        decision.write_text(decision.read_text()+f"| D020 | 2026-11-18 | Select `{selection['selected_model']}` with calibration {'enabled' if selection['calibration_enabled'] else 'disabled'}. | Baseline CV winner `{selection['baseline_winner']}`; TPE best trial {tuning['best_trial']}; evidence: `model_selection.json`, `tuned_vs_untuned.csv`, `calibration_audit.json`. |\n")
    print('Generated Member 2 report, plots and final decision record',flush=True)


if __name__=='__main__':
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results/metrics/member2').glob('*.json')) if not p.name.startswith('trial_') and not p.name.endswith('_progress.json')]
    generate_report(records,json.loads((ROOT/'results/metrics/model_selection.json').read_text()),json.loads((ROOT/'results/metrics/tuning_best.json').read_text()))
