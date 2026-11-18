"""Tune on training trips, freeze, evaluate all student trips, create maps/report."""
from dataclasses import asdict,replace
import itertools
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.models.training import ROOT,load_scorer,file_hash,write_json,scorer_from_fitted
from src.models.calibration import SigmoidCalibrator
from src.evaluation.harness import fit_estimator,_positive_probability
from src.data_processing.loaders import load_trip,trip_directories
from src.hmm.network import RoadNetwork
from src.hmm.viterbi import HMMParams,prepare,decode_prepared,classifier_predictions,route_segments
from src.hmm.predict import predict_trip
from src.evaluation.route_metrics import pointwise_accuracy,route_continuity,sequence_metrics

METRICS=ROOT/'results/metrics'
CACHE=ROOT/'data/processed/member3'


def signature():
    import hashlib
    files=['src/hmm/network.py','src/hmm/viterbi.py','src/hmm/predict.py','src/evaluation/route_metrics.py',
           'models/model.joblib','models/model_card.json','models/feature_schema.json','configs/folds.json',
           'data/processed/train.parquet','data/processed/obs_labels.parquet','results/metrics/data_manifest.csv']
    inputs={name:file_hash(ROOT/name) for name in files}
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),inputs


def tuning_scores(key,inputs):
    score_path=CACHE/'tuning_scores.parquet'
    meta_path=CACHE/'tuning_model.json'
    if score_path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta['signature']==key:
            return pd.read_parquet(score_path),meta
    selected=load_scorer(ROOT/'models')
    features=selected.feature_columns
    frame=pd.read_parquet(ROOT/'data/processed/train.parquet')
    folds=json.loads((ROOT/'configs/folds.json').read_text())['trip_to_fold']
    assignment=frame.trip_id.map(folds)
    fitting=frame[assignment.isin([0,1,2])]
    calibration=frame[assignment==4]
    tuning=frame[assignment==3]
    estimator=clone(selected.estimator)
    if not isinstance(selected.preprocessor,str):
        estimator=Pipeline([('preprocessor',clone(selected.preprocessor)),('estimator',estimator)])
    print('Fit separate decoder-tuning scorer: 48 fit / 16 calibration / 16 tuning trips',flush=True)
    fitted,fit=fit_estimator(estimator,fitting,features)
    calibrator=None
    if selected.calibrator is not None:
        with threadpool_limits(limits=2):
            raw=_positive_probability(fitted,calibration[features])
        known=calibration.is_ref>=0
        calibrator=SigmoidCalibrator().fit(raw[known],calibration.loc[known,'is_ref'])
        if not calibrator.increasing:
            raise ValueError('Tuning scorer calibration must be increasing')
    scorer=scorer_from_fitted(fitted,features,calibrator)
    with threadpool_limits(limits=2):
        scores=scorer.predict_proba(tuning)
    result=tuning[['trip_id','obs_idx','arc_id']].copy()
    result['score']=scores
    result.to_parquet(score_path,index=False)
    meta=dict(signature=key,inputs=inputs,fit=fit,base_fit_trip_ids=sorted(fitting.trip_id.unique()),
              calibration_trip_ids=sorted(calibration.trip_id.unique()),tuning_trip_ids=sorted(tuning.trip_id.unique()),
              calibration_enabled=calibrator is not None,calibration_slope=float(calibrator.model.coef_[0,0]) if calibrator else None,
              limitation='Family/parameters were selected earlier using training CV. This decoder-tuning set is independent of this scorer fit/calibration, not of all prior model selection.')
    write_json(meta_path,meta)
    write_json(METRICS/'hmm_tuning_model.json',meta)
    joblib.dump(scorer,CACHE/'tuning_scorer.joblib',compress=3)
    return result,meta


def prepared_tuning(scores,key,params):
    folder=CACHE/'prepared_tuning'
    folder.mkdir(exist_ok=True)
    prepared=[]
    manifest=pd.read_csv(METRICS/'data_manifest.csv',dtype={'trip_id':str})
    for trip_id,group in scores.groupby('trip_id',sort=True):
        path=folder/f'{trip_id}.joblib'
        if path.exists():
            saved=joblib.load(path)
            if saved['signature']==key:
                prepared.append(saved)
                continue
        print(f'Prepare training transitions {trip_id}',flush=True)
        raw_dir=ROOT/'data/raw/train'/trip_id
        for suffix in ['track','nodes','arcs']:
            expected=manifest[(manifest.partition=='train') & (manifest.trip_id==trip_id) & (manifest.file_type==suffix)].sha256.iloc[0]
            if file_hash(raw_dir/f'{trip_id}.{suffix}')!=expected:
                raise ValueError(f'Changed inference file: {trip_id}.{suffix}')
        started=time.perf_counter()
        network=RoadNetwork.from_trip_dir(raw_dir)
        geometry=prepare(group,network,params,maximum_gap_s=60.)
        saved=dict(signature=key,trip_id=trip_id,prepared=geometry,arcs=network.arcs,
                   prepare_ms=(time.perf_counter()-started)*1000,graph_stats=network.stats())
        joblib.dump(saved,path,compress=3)
        prepared.append(saved)
    return prepared


def tune(prepared,labels,key,inputs):
    results=[];by_trip=[]
    for beta,gap in itertools.product([10.,30.,60.,120.],[10.,30.,60.]):
        params=HMMParams(transition_beta_m=beta,gap_restart_s=gap)
        started=time.perf_counter()
        rows=[]
        for item in prepared:
            trip_id=item['trip_id']
            prediction=decode_prepared(item['prepared'],params)
            prediction['trip_id']=trip_id
            measured=pointwise_accuracy(prediction,labels[labels.trip_id==trip_id])
            continuity=route_continuity(prediction,networks={trip_id:item['arcs']})
            rows.append(dict(trip_id=trip_id,transition_beta_m=beta,gap_restart_s=gap,**measured,
                             route_continuity=continuity,n_obs=len(prediction),n_restarts=prediction.attrs['diagnostics']['n_restarts']))
        count=sum(r['n_known_obs'] for r in rows)
        possible=sum(max(0,r['n_obs']-1) for r in rows)
        summary=dict(transition_beta_m=beta,gap_restart_s=gap,n_known_obs=count,n_correct=sum(r['n_correct'] for r in rows),
                     pointwise_accuracy=sum(r['n_correct'] for r in rows)/count,
                     trip_mean_accuracy=float(np.mean([r['pointwise_accuracy'] for r in rows])),
                     route_continuity=sum(r['route_continuity']*max(0,r['n_obs']-1) for r in rows)/possible,
                     n_restarts=sum(r['n_restarts'] for r in rows),decode_seconds=time.perf_counter()-started)
        results.append(summary);by_trip.extend(rows)
        print(f'HMM beta={beta:g} gap={gap:g}: point accuracy {summary["pointwise_accuracy"]:.4f}',flush=True)
        pd.DataFrame(results).to_csv(METRICS/'hmm_tuning.csv',index=False)
    table=pd.DataFrame(results).sort_values(['pointwise_accuracy','route_continuity','n_restarts','transition_beta_m','gap_restart_s'],ascending=[False,False,True,False,True])
    best=table.iloc[0]
    chosen=HMMParams(transition_beta_m=float(best.transition_beta_m),gap_restart_s=float(best.gap_restart_s))
    pd.DataFrame(by_trip).to_csv(METRICS/'hmm_tuning_by_trip.csv',index=False)
    payload=dict(version='member3-v1',parameters=asdict(chosen),signature=key,inputs=inputs,
                 selection=dict(partition='train fold 3',primary='known-observation pointwise accuracy',
                                tie_breaks=['direct continuity','fewer restarts','larger beta','smaller restart gap'],
                                n_configurations=len(results),n_tuning_trips=len(prepared),n_known_obs=int(best.n_known_obs),
                                tuning_pointwise_accuracy=float(best.pointwise_accuracy),student_test_used=False))
    write_json(ROOT/'configs/hmm.json',payload)
    return chosen,payload


def compare_trip(network,prepared,decoded,params,labels,partition):
    classifier=classifier_predictions(prepared)
    # Ground truth is opened only here, after inference has completed.
    reference=load_trip(ROOT/'data/raw'/('train' if partition=='decoder_tuning' else partition)/network.trip_id,include_route=True).route
    rows=[]
    for variant,prediction,bridge in [('classifier',classifier,False),('classifier_bridged',classifier,True),('hmm',decoded,True)]:
        prediction=prediction.copy()
        prediction['trip_id']=network.trip_id
        segments=route_segments(prediction,network,params,bridge)
        measured=pointwise_accuracy(prediction,labels)
        rows.append(dict(partition=partition,trip_id=network.trip_id,variant=variant,n_obs=len(prediction),**measured,
                         route_continuity=route_continuity(prediction,network),**sequence_metrics(segments,reference),
                         decoder_restarts=decoded.attrs['diagnostics']['n_restarts'] if variant=='hmm' else 0))
    return rows,reference


def aggregate(rows):
    frame=pd.DataFrame(rows)
    output=[]
    for (partition,variant),group in frame.groupby(['partition','variant']):
        known=int(group.n_known_obs.sum());pairs=int((group.n_obs-1).sum())
        overlap=int(group.lcs_arcs.sum());predicted=int(group.predicted_route_arcs.sum());reference=int(group.reference_route_arcs.sum())
        precision=overlap/predicted if predicted else 0
        recall=overlap/reference if reference else 0
        output.append(dict(partition=partition,variant=variant,n_trips=len(group),n_obs=int(group.n_obs.sum()),
                           n_known_obs=known,n_unknown_obs=int(group.n_unknown_obs.sum()),n_correct=int(group.n_correct.sum()),
                           pointwise_accuracy=group.n_correct.sum()/known,trip_mean_accuracy=group.pointwise_accuracy.mean(),
                           route_continuity=float((group.route_continuity*(group.n_obs-1)).sum()/pairs),
                           sequence_precision=precision,sequence_recall=recall,sequence_f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                           trip_mean_sequence_f1=group.sequence_f1.mean(),decoder_restarts=int(group.decoder_restarts.sum())))
    return pd.DataFrame(output)


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    METRICS.mkdir(parents=True,exist_ok=True)
    key,inputs=signature()
    labels=pd.read_parquet(ROOT/'data/processed/obs_labels.parquet')
    scores,model_meta=tuning_scores(key,inputs)
    default=HMMParams()
    prepared=prepared_tuning(scores,key,default)
    params,selection=tune(prepared,labels[labels.partition=='train'],key,inputs)
    comparison=[];timings=[]
    for item in prepared:
        network=RoadNetwork.from_trip_dir(ROOT/'data/raw/train'/item['trip_id'])
        decoded=decode_prepared(item['prepared'],params)
        rows,_=compare_trip(network,item['prepared'],decoded,params,labels[labels.trip_id==item['trip_id']],'decoder_tuning')
        comparison.extend(rows)
    scorer=load_scorer(ROOT/'models')
    from src.visualization.folium_maps import prediction_map
    for folder in trip_directories(ROOT/'data/raw','student_test'):
        print(f'Evaluate student trip {folder.name}',flush=True)
        result=predict_trip(folder,scorer,params)
        rows,reference=compare_trip(result.network,result.prepared,result.decoded,params,labels[labels.trip_id==folder.name],'student_test')
        comparison.extend(rows)
        started=time.perf_counter()
        warm=decode_prepared(result.prepared,params)
        warm_ms=(time.perf_counter()-started)*1000
        timings.append(dict(partition='student_test',trip_id=folder.name,n_obs=len(result.decoded),n_arcs=len(result.network.arcs),
                            **result.timing_ms,warm_viterbi_ms=warm_ms,**result.network.stats()))
        saved=result.classifier.rename(columns={'arc_id':'classifier_arc'}).merge(result.decoded.rename(columns={'arc_id':'hmm_arc'}),on='obs_idx')
        saved.to_parquet(CACHE/f'prediction_{folder.name}.parquet',index=False)
        write_json(CACHE/f'diagnostics_{folder.name}.json',result.decoded.attrs['diagnostics'])
        prediction_map(result,ROOT/f'maps/pred_{folder.name}.html',reference)
        pd.DataFrame(comparison).to_csv(METRICS/'hmm_comparison_by_trip.csv',index=False)
        pd.DataFrame(timings).to_csv(METRICS/'decode_timing.csv',index=False)
    quality=pd.read_csv(METRICS/'quality_report.csv',dtype={'trip_id':str})
    largest=quality[quality.partition=='train'].sort_values('n_obs').iloc[-1].trip_id
    print(f'Measure largest training trip {largest}',flush=True)
    result=predict_trip(ROOT/'data/raw/train'/largest,scorer,params)
    started=time.perf_counter();decode_prepared(result.prepared,params);warm_ms=(time.perf_counter()-started)*1000
    timings.append(dict(partition='train_speed_only',trip_id=largest,n_obs=len(result.decoded),n_arcs=len(result.network.arcs),
                        **result.timing_ms,warm_viterbi_ms=warm_ms,**result.network.stats()))
    # Candidate-cap diagnostic reuses scoring; it does not choose decoder parameters.
    cap=replace(params,candidate_cap=8)
    started=time.perf_counter()
    limited=prepare(result.candidates[['trip_id','obs_idx','arc_id','score']],result.network,cap,maximum_gap_s=cap.gap_restart_s)
    decode_prepared(limited,cap)
    write_json(METRICS/'hmm_cap_diagnostic.json',dict(partition='train_speed_only',trip_id=largest,n_obs=len(result.decoded),
                cap8_warm_graph_prepare_decode_ms=(time.perf_counter()-started)*1000,chosen_candidate_cap=params.candidate_cap,
                note='Warm graph diagnostic only; cap 8 is not selected or used in student predictions.'))
    pd.DataFrame(timings).to_csv(METRICS/'decode_timing.csv',index=False)
    summary=aggregate(comparison)
    summary.to_csv(METRICS/'hmm_comparison.csv',index=False)
    write_json(METRICS/'member3_run.json',dict(signature=key,inputs=inputs,parameters=asdict(params),n_student_trips=10,
               n_tuning_trips=len(prepared),selected_model_checksum=file_hash(ROOT/'models/model.joblib'),
               provisional_cold_student_budget_ms=10000))
    from src.hmm.member3_report import generate_report
    generate_report()
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
