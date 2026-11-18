"""Run/reuse five baselines, tune the CV winner, audit calibration, export/report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.evaluation.harness import evaluate_model,evaluate_holdout,score_predictions,_positive_probability,fit_estimator
from src.evaluation.continuity import route_continuity
from src.models.specs import BASELINES,make_estimator
from src.models.training import ROOT,load_training_data,input_signature,write_json,file_hash,export_scorer,scorer_from_fitted,CandidateScorer,load_scorer

METRICS=ROOT/'results/metrics'
CACHE=ROOT/'data/processed/member2'


def _cached(path,signature):
    if Path(path).exists():
        result=json.loads(Path(path).read_text())
        if result.get('signature')==signature:
            return result
    return None


def run_baseline(name,frame,student,labels,features,seed):
    estimator=make_estimator(name,seed=seed)
    signature,inputs=input_signature(features,estimator)
    path=METRICS/f'member2/{name}.json'
    cached=_cached(path,signature)
    folder=ROOT/f'models/member2/{name}'
    if cached and (folder/'model_card.json').exists() and (CACHE/f'{name}_oof.npy').exists():
        load_scorer(folder)
        print(f'{name}: reuse matching completed run',flush=True)
        return cached
    cv=evaluate_model(estimator,name,frame,features,obs_labels=labels[labels.partition=='train'],
                      continuity_fn=route_continuity,output_path=METRICS/f'member2/{name}_progress.json',
                      oof_path=CACHE/f'{name}_oof.npy')
    test,fitted,scores=evaluate_holdout(estimator,frame,student,labels[labels.partition=='student_test'],features,
                                      model_name=name,continuity_fn=route_continuity,return_estimator=True)
    diagnostic=frame[frame.is_ref>=0].sample(n=min(10000,int((frame.is_ref>=0).sum())),random_state=seed)
    with threadpool_limits(limits=2):
        dp=_positive_probability(fitted,diagnostic[features])
    from sklearn.metrics import f1_score
    record=dict(signature=signature,inputs=inputs,name=name,cv=cv,student=test,
                training_diagnostic_macro_f1=float(f1_score(diagnostic.is_ref,dp>=.5,labels=[0,1],average='macro',zero_division=0)),
                diagnostic_scope='<=10k fitted rows; optimistic, not validation',configuration=estimator.get_params())
    scorer=scorer_from_fitted(fitted,features)
    card=export_scorer(scorer,name,features,folder,dict(status='baseline',seed=seed,fit={k:test[k] for k in ['n_fit_rows','n_fit_trips','train_time_sec','warnings']},signature=signature,inputs=inputs,metrics=test))
    np.save(CACHE/f'{name}_student.npy',scores)
    write_json(path,record)
    if name=='logistic_regression' and not (ROOT/'models/model_card.json').exists():
        export_scorer(scorer,name,features,ROOT/'models',dict(card,status='provisional'))
        print('Provisional Logistic Regression exported to models/',flush=True)
    return record


def import_member1(name,frame,student,labels,features):
    from src.models.member1_harness import load_scorer as load_member1
    source=ROOT/f'models/member1/{name}'
    card=json.loads((source/'model_card.json').read_text())
    if card['max_train_rows']!=0:
        raise ValueError('Comparison requires full known-row Member 1 experiments')
    for filename,expected in card['input_sha256'].items():
        path=Path(filename)
        if not path.is_absolute():
            path=ROOT/path
        if not path.exists() and path.name=='member1_harness.py':
            path=ROOT/'src/models/member1_harness.py'
        if file_hash(path)!=expected:
            raise ValueError(f'Stale Member 1 input: {filename}')
    signature,inputs=input_signature(features,extra=dict(member1=name,card_hash=file_hash(source/'model_card.json')))
    path=METRICS/f'member2/{name}.json'
    cached=_cached(path,signature)
    if cached:
        return cached
    rows=json.loads((ROOT/f'results/metrics/{name}_experiments.json').read_text())
    folds=[]
    for row in rows:
        if row['partition']!='cv':
            continue
        row=dict(row,top1_accuracy=row['top1_acc'],train_time_sec=row['train_time_s'],inference_time_sec=row['inference_time_s'],warnings=row['fit_warnings'])
        folds.append(row)
    if len(folds)!=5 or any(row['n_fit_trips']!=64 for row in folds):
        raise ValueError('Member 1 CV is not the expected five trip folds')
    keys=['macro_f1','log_loss','top1_accuracy','precision','recall','f1','candidate_accuracy','candidate_miss_rate','train_time_sec','inference_time_sec']
    cv={k:float(np.mean([r[k] for r in folds])) for k in keys}
    cv.update(folds=folds,macro_f1_std=card['cv_macro_f1_std'],route_continuity=None,brier_score=None,ece_10=None,
              note='Historical CV probabilities were not retained; continuity/calibration CV values are unavailable.')
    print(f'{name}: score saved Member 1 model for continuity/calibration diagnostics',flush=True)
    scorer=load_member1(source)
    started=time.perf_counter()
    with threadpool_limits(limits=2):
        scores=scorer.predict_proba(student)
    inference=time.perf_counter()-started
    test=score_predictions(student,scores,labels[labels.partition=='student_test'],route_continuity)
    test.update(inference_time_sec=inference,train_time_sec=card['fit']['train_time_s'],n_fit_rows=card['fit']['n_fit_rows'],n_fit_trips=80,warnings=card['fit']['fit_warnings'])
    record=dict(name=name,signature=signature,inputs=inputs,cv=cv,student=test,configuration=card['configuration'],
                training_diagnostic_macro_f1=card['training_diagnostic_macro_f1'],diagnostic_scope=card['training_diagnostic_scope'],
                member1_source=str(source.relative_to(ROOT)))
    np.save(CACHE/f'{name}_student.npy',scores)
    write_json(path,record)
    return record


def comparison_table(records):
    rows=[]
    for record in records:
        row=dict(model_name=record['name'],configuration=json.dumps(record['configuration'],default=str),training_diagnostic_macro_f1=record.get('training_diagnostic_macro_f1'))
        for prefix,key in [('cv','cv'),('student','student')]:
            for metric in ['candidate_accuracy','precision','recall','f1','macro_f1','macro_f1_std',
                           'top1_accuracy','trip_mean_top1','candidate_miss_rate','route_continuity',
                           'log_loss','brier_score','ece_10','train_time_sec','inference_time_sec',
                           'n_known_obs','n_unknown_obs','n_candidate_rows','n_known_candidate_rows']:
                if metric in record[key]:
                    row[f'{prefix}_{metric}']=record[key][metric]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['cv_top1_accuracy','cv_macro_f1','model_name'],ascending=[False,False,True])



def flatten_experiments(records):
    """Normalize reserved fields after importing legacy experiment records."""
    rows=[]
    for record in records:
        for fold in record['cv']['folds']:
            scalar={k:v for k,v in fold.items() if v is None or isinstance(v,(str,int,float))}
            rows.append(dict(scalar,model_name=record['name'],partition='cv'))
        scalar={k:v for k,v in record['student'].items() if v is None or isinstance(v,(str,int,float))}
        rows.append(dict(scalar,model_name=record['name'],partition='student_test'))
    return rows


def select_winner(table):
    eligible=table.dropna(subset=['cv_top1_accuracy','cv_macro_f1'])
    if eligible.empty:
        raise ValueError('No CV-complete model')
    return eligible.sort_values(['cv_top1_accuracy','cv_macro_f1','model_name'],ascending=[False,False,True]).iloc[0].model_name


def run(data_path=None,seed=42,output_dir=None,student_path=None,trials=10,baseline_only=False):
    CACHE.mkdir(parents=True,exist_ok=True)
    (METRICS/'member2').mkdir(parents=True,exist_ok=True)
    frame,features=load_training_data(data_path) if data_path else load_training_data()
    student=pd.read_parquet(student_path or ROOT/'data/processed/student_test.parquet')
    labels=pd.read_parquet(ROOT/'data/processed/obs_labels.parquet')
    if set(frame.trip_id)&set(student.trip_id):
        raise ValueError('Training/student trips overlap')
    for partition,table,expected in [('train',frame,80),('student_test',student,10)]:
        reference=labels[labels.partition==partition]
        if table.trip_id.nunique()!=expected or set(table.trip_id)!=set(reference.trip_id):
            raise ValueError(f'Unexpected trip population: {partition}')
    manifest=pd.read_csv(METRICS/'data_manifest.csv',dtype={'trip_id':str})
    for row in manifest[manifest.file_type=='arcs'].itertuples():
        if row.partition not in {'train','student_test'}:
            raise ValueError('Unexpected road-network partition')
        network=ROOT/'data/raw'/row.partition/row.trip_id/f'{row.trip_id}.arcs'
        if file_hash(network)!=row.sha256:
            raise ValueError(f'Changed road network: {row.trip_id}')
    records=[]
    for name in BASELINES:
        records.append(run_baseline(name,frame,student,labels,features,seed))
    for name in ['knn','mlp']:
        records.append(import_member1(name,frame,student,labels,features))
    table=comparison_table(records)
    table.to_csv(METRICS/'model_comparison.csv',index=False)
    winner=select_winner(table)
    write_json(METRICS/'model_selection.json',dict(baseline_winner=winner,metric='mean five-fold observation Top-1',student_test_used_for_selection=False))
    print(f'Baseline CV winner: {winner}',flush=True)
    if baseline_only:
        return table
    from src.models.tuning.bayes_opt import tune
    tuning,tuned_cv=tune(trials=trials,seed=seed,model_name=winner,frame=frame,features=features,obs_labels=labels[labels.partition=='train'])
    estimator=make_estimator(winner,tuning['best_params'],seed)
    tuned_name=winner+'_tuned'
    tuned_signature,tuned_inputs=input_signature(features,estimator)
    tuned_path=METRICS/f'member2/{tuned_name}.json'
    tuned=_cached(tuned_path,tuned_signature)
    if tuned is None:
        test,fitted,scores=evaluate_holdout(estimator,frame,student,labels[labels.partition=='student_test'],features,
                                          model_name=tuned_name,continuity_fn=route_continuity,return_estimator=True)
        tuned=dict(name=tuned_name,signature=tuned_signature,inputs=tuned_inputs,configuration=estimator.get_params(),cv=tuned_cv,student=test)
        export_scorer(scorer_from_fitted(fitted,features),tuned_name,features,ROOT/f'models/member2/{tuned_name}',dict(status='tuned_candidate',seed=seed,fit=test,signature=tuned_signature,inputs=tuned_inputs))
        np.save(CACHE/f'{tuned_name}_student.npy',scores)
        write_json(tuned_path,tuned)
    baseline=next(r for r in records if r['name']==winner)
    use_tuned=tuned['cv']['top1_accuracy']>baseline['cv']['top1_accuracy']
    selected=tuned if use_tuned else baseline
    selected_estimator=estimator if use_tuned else make_estimator(winner,seed=seed)
    records.append(tuned)
    table=comparison_table(records)
    table.to_csv(METRICS/'model_comparison.csv',index=False)
    comparison_table([baseline,tuned]).to_csv(METRICS/'tuned_vs_untuned.csv',index=False)
    from src.models.calibration import audit_calibration,fit_oof_calibrator
    calibration_accepted,audit=audit_calibration(selected_estimator,frame,features,labels[labels.partition=='train'])
    calibrator=None
    if calibration_accepted:
        if use_tuned:
            oof=np.load(ROOT/tuning['oof_path'])
        elif (CACHE/f'{winner}_oof.npy').exists():
            oof=np.load(CACHE/f'{winner}_oof.npy')
        else:
            evaluate_model(selected_estimator,winner+' calibration OOF',frame,features,obs_labels=labels[labels.partition=='train'],oof_path=CACHE/f'{winner}_oof.npy',continuity_fn=route_continuity)
            oof=np.load(CACHE/f'{winner}_oof.npy')
        calibrator=fit_oof_calibrator(frame,oof)
    if selected['name'] in ['knn','mlp']:
        source=ROOT/f'models/member1/{winner}'
        scorer=CandidateScorer(joblib.load(source/'model.joblib'),joblib.load(source/'preprocessor.joblib'),features,calibrator)
    else:
        scorer=load_scorer(ROOT/f'models/member2/{selected["name"]}')
        scorer.calibrator=calibrator
    raw_scores=np.load(CACHE/f'{selected["name"]}_student.npy')
    if calibrator is None:
        final_scores=np.clip(raw_scores,np.finfo(float).eps,1-np.finfo(float).eps)
    else:
        final_scores=np.clip(calibrator.predict(raw_scores),np.finfo(float).eps,1-np.finfo(float).eps)
    final_metrics=score_predictions(student,final_scores,labels[labels.partition=='student_test'],route_continuity)
    selection=dict(baseline_winner=winner,selected_model=selected['name'],tuned_selected=bool(use_tuned),calibration_enabled=calibration_accepted,
                   metric='mean observation Top-1 across frozen five trip folds; strict CV improvement required to replace baseline',
                   student_test_used_for_selection=False,cv_top1_accuracy=selected['cv']['top1_accuracy'],cv_macro_f1=selected['cv']['macro_f1'],
                   final_student_test=final_metrics,
                   limitations=['Known-label surrogate alignment; unknown observations excluded from supervised metrics',
                                'CV conditional on training-wide candidate selection and model/search selection; not nested CV',
                                'Continuity is direct classifier adjacency; C1/HMM/sequence comparison remains Member 3 work'])
    signature,inputs=input_signature(features,selected_estimator,extra=dict(calibrated=calibration_accepted))
    card=export_scorer(scorer,selected['name'],features,output_dir or ROOT/'models',
                       dict(status='final',seed=seed,selection=selection,inputs=inputs,signature=signature,train_rows=int((frame.is_ref>=0).sum()),train_trips=80,
                            calibration=dict(enabled=calibration_accepted,method='OOF sigmoid on log odds' if calibration_accepted else 'none',audit='results/metrics/calibration_audit.json',
                                             slope=float(calibrator.model.coef_[0,0]) if calibrator else None),metrics=final_metrics))
    write_json(METRICS/'model_selection.json',selection)
    np.save(CACHE/'final_student.npy',final_scores)
    pd.DataFrame(flatten_experiments(records)).to_csv(METRICS/'experiments.csv',index=False)
    from src.models.member2_report import generate_report
    generate_report(records,selection,tuning)
    return table


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-dir')
    parser.add_argument('--student-test')
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args()
    print(run(args.data,args.seed,args.output_dir,args.student_test,args.trials,args.baseline_only).to_string(index=False))
