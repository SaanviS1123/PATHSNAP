import itertools
import numpy as np
import pandas as pd
import pytest
from src.data_processing.loaders import Trip
from src.data_processing.geo import project_trip
from src.hmm.network import RoadNetwork
from src.hmm.viterbi import HMMParams,PreparedTrip,prepare,decode,decode_prepared,route_segments
from src.evaluation.route_metrics import lcs_length,sequence_metrics,pointwise_accuracy,route_continuity


def network(times=(0,1,2)):
    nodes=np.array([[0.,0.],[.0001,0.],[.0002,0.],[.0001,.0001],[.0002,.0001]])
    arcs=np.array([[0,1],[1,2],[1,3],[3,4]])
    track=np.array([[.00005,0,times[0]],[.00012,0,times[1]],[.00018,0,times[2]]])
    return RoadNetwork(project_trip(Trip('00000001',track,nodes,arcs)))


def candidates():
    return pd.DataFrame(dict(trip_id=['00000001']*4,obs_idx=[0,1,1,2],arc_id=[0,1,2,1],score=[.9,.2,.95,.9]))


def test_viterbi_uses_future_connectivity_instead_of_greedy():
    road=network()
    prediction=decode(candidates(),road)
    assert prediction.arc_id.tolist()==[0,1,1]
    assert prediction.attrs['diagnostics']['n_restarts']==0
    assert route_segments(prediction,road)==[[0,1]]


def test_partial_arc_distances_direction_and_bounded_parallel_edges():
    road=network()
    distance=road.transition_distances(np.array([0]),np.array([1]),np.array([.5]),np.array([.5]),100)[0,0]
    assert distance==pytest.approx(.5*road.lengths[0]+.5*road.lengths[1])
    assert not np.isfinite(road.shortest_path(2,0,100)[0])
    metric=road.metric
    trip=Trip('00000001',metric.trip.track,metric.trip.nodes,np.array([[0,1],[0,1],[1,2]]))
    duplicate=RoadNetwork(project_trip(trip))
    length,path=duplicate.shortest_path(0,1,100)
    assert length==pytest.approx(duplicate.lengths[0]) and path==[0]
    assert not np.isfinite(duplicate.shortest_path(0,1,1)[0])
    duplicate.shortest_path(0,1,100)
    assert duplicate.stats()['cache_hits']>0


def test_gap_unreachable_and_empty_observations_restart_explicitly():
    road=network(times=(0,100,101))
    prediction=decode(candidates(),road,HMMParams(gap_restart_s=30))
    assert prediction.attrs['diagnostics']['restarts']['1']=='time_gap'
    only=pd.DataFrame(dict(trip_id=['00000001']*2,obs_idx=[0,1],arc_id=[1,3],score=[.9,.9]))
    prediction=decode(only,network())
    assert prediction.arc_id.tolist()==[1,3,-1]
    assert prediction.attrs['diagnostics']['restarts']['1']=='unreachable'
    assert route_segments(prediction,network())==[[1],[3]]
    empty=candidates().iloc[:0]
    assert decode(empty,network()).arc_id.tolist()==[-1,-1,-1]


def test_backward_jitter_tolerance_and_reference_fields_rejected():
    road=network()
    small=road.transition_distances(np.array([0]),np.array([0]),np.array([.6]),np.array([.4]),100,backtrack_m=3)
    blocked=road.transition_distances(np.array([0]),np.array([0]),np.array([.9]),np.array([.1]),100,backtrack_m=3)
    assert small[0,0]==0 and not np.isfinite(blocked[0,0])
    with pytest.raises(ValueError,match='Reference fields'):
        decode(candidates().assign(is_ref=0),road)
    with pytest.raises(ValueError,match='finite'):
        HMMParams(transition_beta_m=float('nan'))
    malformed=candidates().astype({'obs_idx':'float64'})
    malformed.loc[0,'obs_idx']=.5
    with pytest.raises(ValueError,match='integers'):
        decode(malformed,road)


def test_decoder_matches_exhaustive_log_score_on_tiny_chain():
    arcs=np.array([[0,1],[0,1],[0,1]],dtype=np.int32)
    probabilities=np.array([[.6,.4],[.3,.7],[.8,.2]])
    distances=np.array([[[1.,3.],[2.,1.]],[[1.,2.],[3.,1.]]])
    prepared=PreparedTrip('00000001',arcs,np.zeros_like(arcs,dtype=float),np.log(probabilities),distances,np.ones(2),np.ones(2),np.full(3,2))
    params=HMMParams(transition_beta_m=2)
    def score(path):
        return sum(np.log(probabilities[t,path[t]]) for t in range(3))-sum(abs(distances[t,path[t],path[t+1]]-1)/2 for t in range(2))
    expected=max(itertools.product(range(2),repeat=3),key=score)
    assert decode_prepared(prepared,params).arc_id.tolist()==list(expected)


def test_metrics_preserve_order_occurrences_and_unknown_denominators():
    assert lcs_length([1,2,1],[1,1,2])==2
    result=sequence_metrics([[1,2,1]],[1,1,2])
    assert result['sequence_f1']==pytest.approx(2/3)
    assert not result['exact_route_match']
    assert sequence_metrics([],[])['sequence_f1']==1
    labels=pd.DataFrame(dict(trip_id=['x']*3,obs_idx=[0,1,2],ref_arc=[0,1,-1]))
    pred=pd.DataFrame(dict(trip_id=['x']*3,obs_idx=[0,1,2],arc_id=[0,-1,1]))
    assert pointwise_accuracy(pred,labels)['pointwise_accuracy']==.5
    assert route_continuity(pred,networks={'x':np.array([[0,1],[1,2]])})==0


def test_inference_loader_never_opens_reference(tmp_path,monkeypatch):
    from src.data_processing.synthetic import write_fixture
    write_fixture(tmp_path)
    original=np.loadtxt
    def guarded(path,*args,**kwargs):
        assert not str(path).endswith('.route')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'loadtxt',guarded)
    road=RoadNetwork.from_trip_dir(tmp_path/'00000001')
    assert road.metric.trip.route is None
