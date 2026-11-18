"""Member 3 inference entry point. Reference data is never loaded here."""
from dataclasses import dataclass
import json
from pathlib import Path
import time
from threadpoolctl import threadpool_limits
from src.features.build import build_features
from src.hmm.network import RoadNetwork
from src.hmm.viterbi import HMMParams,prepare,decode_prepared,classifier_predictions,route_segments
from src.models.training import load_scorer,ROOT


@dataclass
class TripPrediction:
    network: RoadNetwork
    candidates: object
    prepared: object
    classifier: object
    decoded: object
    segments: list
    timing_ms: dict


def load_params(path=None):
    payload=json.loads(Path(path or ROOT/'configs/hmm.json').read_text())
    return HMMParams(**payload.get('parameters',payload))


def predict_trip(trip_dir,scorer=None,params=None):
    started=time.perf_counter()
    params=params or load_params()
    scorer=scorer or load_scorer(ROOT/'models')
    network=RoadNetwork.from_trip_dir(trip_dir)
    loaded=time.perf_counter()
    config=json.loads((ROOT/'configs/candidates.json').read_text())
    candidates=build_features(network.metric,config['radius_m'],config['k'])
    featured=time.perf_counter()
    with threadpool_limits(limits=2):
        candidates['score']=scorer.predict_proba(candidates)
    scored=time.perf_counter()
    prepared=prepare(candidates[['trip_id','obs_idx','arc_id','score']],network,params,maximum_gap_s=params.gap_restart_s)
    routed=time.perf_counter()
    classifier=classifier_predictions(prepared)
    decoded=decode_prepared(prepared,params)
    decoded_at=time.perf_counter()
    segments=route_segments(decoded,network,params,bridge=True)
    finished=time.perf_counter()
    timing=dict(load_project_ms=(loaded-started)*1000,feature_ms=(featured-loaded)*1000,score_ms=(scored-featured)*1000,
                transition_prepare_ms=(routed-scored)*1000,viterbi_ms=(decoded_at-routed)*1000,
                route_expand_ms=(finished-decoded_at)*1000,total_ms=(finished-started)*1000)
    return TripPrediction(network,candidates,prepared,classifier,decoded,segments,timing)
