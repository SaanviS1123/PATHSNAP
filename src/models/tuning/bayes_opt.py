"""Persistent, seeded TPE optimisation of the observed baseline CV winner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
from src.evaluation.harness import evaluate_model,make_group_splits,fit_estimator,_positive_probability,score_predictions
from src.models.member1_harness import training_sample
from src.evaluation.continuity import route_continuity
from src.models.specs import make_estimator,suggest_params,decode_trial_params
from src.models.training import ROOT,load_training_data,input_signature,file_hash,write_json


def tune(data_path=None,trials=10,seed=42,output_path=None,model_name=None,frame=None,features=None,obs_labels=None,search_max_train_rows=250000):
    import optuna
    if trials<4:
        raise ValueError('At least four trials are required: three startup plus a TPE proposal')
    if frame is None:
        frame,features=load_training_data(data_path) if data_path else load_training_data()
    if obs_labels is None:
        obs_labels=pd.read_parquet(ROOT/'data/processed/obs_labels.parquet')
        obs_labels=obs_labels[obs_labels.partition=='train']
    if model_name is None:
        selection=json.loads((ROOT/'results/metrics/model_selection.json').read_text())
        model_name=selection['baseline_winner']
    signature,inputs=input_signature(features,extra=dict(family=model_name,seed=seed,search_max_train_rows=search_max_train_rows,search_source=file_hash(__file__)))
    cache=ROOT/'data/processed/member2'
    cache.mkdir(parents=True,exist_ok=True)
    records=ROOT/'results/metrics/member2'
    records.mkdir(parents=True,exist_ok=True)
    study_name=f'{model_name}-{signature[:16]}'
    sampler_path=cache/f'{study_name}_sampler.joblib'
    sampler=joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=seed,n_startup_trials=3)
    study=optuna.create_study(direction='maximize',sampler=sampler,study_name=study_name,
                             storage=f'sqlite:///{cache / (study_name+".sqlite3")}',load_if_exists=True)
    started=time.perf_counter()

    def objective(trial):
        params=suggest_params(trial,model_name)
        estimator=make_estimator(model_name,params,seed)
        trial_started=time.perf_counter()
        folds=[]
        for fold,(train_idx,val_idx) in enumerate(make_group_splits(frame.trip_id)):
            print(f'{model_name} TPE trial {trial.number}: fold {fold}',flush=True)
            training=training_sample(frame.iloc[train_idx],search_max_train_rows)
            validation=frame.iloc[val_idx]
            fitted,fit=fit_estimator(estimator,training,features)
            inference_started=time.perf_counter()
            with threadpool_limits(limits=2):
                scores=_positive_probability(fitted,validation[features])
            inference_seconds=time.perf_counter()-inference_started
            validation_labels=obs_labels[obs_labels.trip_id.isin(validation.trip_id.unique())]
            measured=score_predictions(validation,scores,validation_labels,route_continuity)
            folds.append(dict(fold=fold,**measured,**fit,inference_time_sec=inference_seconds))
            write_json(records/f'trial_{trial.number}_progress.json',dict(folds=folds,training_row_budget=search_max_train_rows))
        keys=['macro_f1','top1_accuracy','log_loss','brier_score','ece_10','route_continuity','candidate_miss_rate']
        result={key:float(np.mean([r[key] for r in folds])) for key in keys}
        result.update(folds=folds,training_row_budget=search_max_train_rows)
        result.update(signature=signature,trial=trial.number,params=params,elapsed_s=time.perf_counter()-trial_started)
        write_json(records/f'trial_{trial.number}.json',result)
        trial.set_user_attr('elapsed_s',result['elapsed_s'])
        trial.set_user_attr('macro_f1',result['macro_f1'])
        trial.set_user_attr('log_loss',result['log_loss'])
        return result['top1_accuracy']

    def persist(study,trial):
        joblib.dump(study.sampler,sampler_path)
        rows=[]
        for item in study.trials:
            rows.append(dict(trial=item.number,state=item.state.name,objective_top1_accuracy=item.value,
                             proposal='startup' if item.number<3 else 'TPE',params_json=json.dumps(item.params),
                             elapsed_s=item.user_attrs.get('elapsed_s'),macro_f1=item.user_attrs.get('macro_f1'),
                             log_loss=item.user_attrs.get('log_loss')))
        pd.DataFrame(rows).to_csv(ROOT/'results/metrics/tuning_trials.csv',index=False)

    study.optimize(objective,n_trials=max(0,trials-len(study.trials)),callbacks=[persist])
    persist(study,None)
    best=study.best_trial
    result=json.loads((records/f'trial_{best.number}.json').read_text())
    confirmation_path=ROOT/'results/metrics/tuning_confirmation.json'
    confirmation_oof=cache/f'{study_name}_confirmation_oof.npy'
    confirmation_key=dict(signature=signature,params=decode_trial_params(model_name,best.params))
    confirmation=None
    if confirmation_path.exists() and confirmation_oof.exists():
        cached=json.loads(confirmation_path.read_text())
        if cached.get('confirmation_key')==confirmation_key:
            confirmation=cached
    if confirmation is None:
        confirmation=evaluate_model(make_estimator(model_name,decode_trial_params(model_name,best.params),seed),model_name+' full-row tuning confirmation',frame,features,obs_labels=obs_labels,continuity_fn=route_continuity,oof_path=confirmation_oof,output_path=ROOT/'results/metrics/member2/tuning_confirmation_progress.json')
        confirmation.update(confirmation_key=confirmation_key)
        write_json(confirmation_path,confirmation)
    summary=dict(model_name=model_name,objective='mean observation Top-1 across all five frozen trip folds, with budgeted training rows and complete validation folds',
                 n_trials=len(study.trials),n_startup_trials=3,n_tpe_trials=sum(t.number>=3 for t in study.trials),
                 best_trial=best.number,best_top1_accuracy=best.value,search_max_train_rows=search_max_train_rows,full_cv_top1_accuracy=confirmation['top1_accuracy'],full_cv_macro_f1=confirmation['macro_f1'],
                 best_params=decode_trial_params(model_name,best.params),sampler='Optuna TPESampler',seed=seed,
                 total_compute_seconds=sum(t.user_attrs.get('elapsed_s',0) for t in study.trials),
                 current_invocation_seconds=time.perf_counter()-started,signature=signature,inputs=inputs,
                 search_space={k:str(v) for k,v in best.distributions.items()},
                 oof_path=str(confirmation_oof.relative_to(ROOT)))
    write_json(ROOT/'results/metrics/tuning_best.json',summary)
    return summary,confirmation


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--model',dest='model_name')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    print(tune(trials=args.trials,seed=args.seed,model_name=args.model)[0])
