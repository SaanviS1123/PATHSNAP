"""Log-space Viterbi over varying directed candidate arcs, with explicit restarts."""
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
from src.data_processing.geo import segment_projection


@dataclass(frozen=True)
class HMMParams:
    transition_beta_m: float=30.
    gap_restart_s: float=30.
    path_margin_m: float=500.
    max_route_distance_m: float=5000.
    backtrack_tolerance_m: float=20.
    candidate_cap: int=16

    def __post_init__(self):
        if not np.isfinite([self.transition_beta_m,self.gap_restart_s,self.path_margin_m,self.max_route_distance_m,self.backtrack_tolerance_m]).all() or not isinstance(self.candidate_cap,int):
            raise ValueError('HMM parameters must be finite; candidate_cap must be an integer')
        if self.transition_beta_m<=0 or self.gap_restart_s<=0 or self.path_margin_m<=0 or self.max_route_distance_m<=0 or self.backtrack_tolerance_m<0 or self.candidate_cap<1:
            raise ValueError('Invalid HMM parameters')


@dataclass
class PreparedTrip:
    trip_id: str
    arcs: np.ndarray
    fractions: np.ndarray
    emissions: np.ndarray
    distances: np.ndarray
    displacement: np.ndarray
    dt: np.ndarray
    available: np.ndarray


def prepare(candidates,network,params=HMMParams(),maximum_gap_s=60.):
    required={'trip_id','obs_idx','arc_id','score'}
    if not required.issubset(candidates):
        raise ValueError('Candidates require trip_id, obs_idx, arc_id, score')
    if {'is_ref','ref_arc','route_position'}&set(candidates):
        raise ValueError('Reference fields must not enter the decoder')
    if not candidates.empty and set(candidates.trip_id.astype(str))!={network.trip_id}:
        raise ValueError('Candidate trip/network mismatch')
    if candidates.duplicated(['obs_idx','arc_id']).any():
        raise ValueError('Duplicate candidates')
    scores=candidates.score.to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Scores must be finite probabilities')
    for column in ['obs_idx','arc_id']:
        ids=candidates[column].to_numpy(dtype=float)
        if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all():
            raise ValueError('Candidate IDs must be finite integers')
    n=len(network.xy)
    if len(candidates) and ((candidates.obs_idx<0)|(candidates.obs_idx>=n)|(candidates.arc_id<0)|(candidates.arc_id>=len(network.arcs))).any():
        raise ValueError('Candidate index outside trip/network')
    # Preserve all schema candidates, ordered by arc ID for deterministic DP ties.
    ranked=candidates.sort_values(['obs_idx','score','arc_id'],ascending=[True,False,True])
    ranked=ranked.groupby('obs_idx',sort=False).head(params.candidate_cap).sort_values(['obs_idx','arc_id'])
    k=max(1,int(ranked.groupby('obs_idx').size().max())) if len(ranked) else 1
    arcs=np.full((n,k),-1,dtype=np.int32)
    fractions=np.zeros((n,k),dtype=np.float64)
    emissions=np.full((n,k),-np.inf,dtype=np.float64)
    available=np.zeros(n,dtype=np.int32)
    for obs,group in ranked.groupby('obs_idx',sort=False):
        obs=int(obs)
        ids=group.arc_id.to_numpy(dtype=int)
        count=len(ids)
        arcs[obs,:count]=ids
        _,position=segment_projection(network.xy[obs],network.metric.start[ids],network.metric.vector[ids])
        fractions[obs,:count]=position
        p=np.clip(group.score.to_numpy(),np.finfo(float).eps,1.)
        emissions[obs,:count]=np.log(p/p.sum())
        available[obs]=count
    displacement=np.linalg.norm(np.diff(network.xy,axis=0),axis=1)
    dt=np.diff(network.timestamps)
    distances=np.full((max(0,n-1),k,k),np.inf,dtype=np.float32)
    for step in range(n-1):
        a,b=available[step:step+2]
        if a==0 or b==0 or dt[step]<=0 or dt[step]>maximum_gap_s or displacement[step]>params.max_route_distance_m:
            continue
        cutoff=min(params.max_route_distance_m,float(displacement[step])+params.path_margin_m)
        distances[step,:a,:b]=network.transition_distances(arcs[step,:a],arcs[step+1,:b],fractions[step,:a],fractions[step+1,:b],cutoff,params.backtrack_tolerance_m)
    return PreparedTrip(network.trip_id,arcs,fractions,emissions,distances,displacement,dt,available)


def decode_prepared(prepared,params=HMMParams()):
    n,k=prepared.arcs.shape
    chosen=np.full(n,-1,dtype=np.int32)
    parent=np.full((n,k),-1,dtype=np.int16)
    delta=np.full((n,k),-np.inf)
    restarts=np.zeros(n,dtype=bool)
    reasons={}
    for t in range(n):
        if prepared.available[t]==0:
            reasons[t]='empty_candidates'
            restarts[t]=True
            continue
        reason=None
        if t==0:
            reason='start'
        elif prepared.available[t-1]==0:
            reason='after_empty_candidates'
        elif prepared.dt[t-1]<=0 or prepared.dt[t-1]>params.gap_restart_s:
            reason='time_gap'
        elif prepared.displacement[t-1]>params.max_route_distance_m:
            reason='spatial_jump'
        if reason is None:
            transition=-np.abs(prepared.distances[t-1]-prepared.displacement[t-1])/params.transition_beta_m
            proposal=delta[t-1,:,None]+transition
            predecessor=np.argmax(proposal,axis=0)
            best=proposal[predecessor,np.arange(k)]
            if not np.isfinite(best).any():
                reason='unreachable'
            else:
                delta[t]=best+prepared.emissions[t]
                parent[t]=predecessor
        if reason is not None:
            delta[t]=prepared.emissions[t]
            restarts[t]=True
            reasons[t]=reason
        if np.isfinite(delta[t]).any():
            delta[t]-=np.max(delta[t])  # Common offset preserves the recurrence.
    # Each restart terminates the preceding independent chain.
    boundaries=list(np.flatnonzero(restarts))+[n]
    starts=sorted(set([0,*boundaries[:-1]]))
    for start,end in zip(starts,starts[1:]+[n]):
        if start==end or prepared.available[start]==0:
            continue
        state=int(np.argmax(delta[end-1]))
        for t in range(end-1,start-1,-1):
            chosen[t]=prepared.arcs[t,state]
            state=int(parent[t,state])
            if t>start and state<0:
                raise RuntimeError('Broken Viterbi backpointer')
    result=pd.DataFrame({'obs_idx':np.arange(n),'arc_id':chosen})
    result.attrs['diagnostics']=dict(n_observations=n,empty_observations=int((prepared.available==0).sum()),
                                     restarts={str(key):value for key,value in reasons.items()},
                                     n_restarts=sum(value not in {'start','empty_candidates'} for value in reasons.values()),params=asdict(params))
    return result


def decode(cands,network,params=HMMParams()):
    return decode_prepared(prepare(cands,network,params,maximum_gap_s=params.gap_restart_s),params)


def classifier_predictions(prepared):
    states=np.argmax(prepared.emissions,axis=1)
    arcs=prepared.arcs[np.arange(len(states)),states].copy()
    arcs[prepared.available==0]=-1
    return pd.DataFrame({'obs_idx':np.arange(len(arcs)),'arc_id':arcs})


def route_segments(predictions,network,params=HMMParams(),bridge=True):
    segments=[]
    previous=None
    for row in predictions.sort_values('obs_idx').itertuples():
        obs,arc=int(row.obs_idx),int(row.arc_id)
        if arc<0:
            previous=None
            continue
        reset=previous is None
        addition=[]
        if previous is not None:
            old_obs,old_arc,old_fraction=previous
            dt=network.timestamps[obs]-network.timestamps[old_obs]
            displacement=float(np.linalg.norm(network.xy[obs]-network.xy[old_obs]))
            reset=obs!=old_obs+1 or dt<=0 or dt>params.gap_restart_s or displacement>params.max_route_distance_m
        _,fraction=segment_projection(network.xy[obs],network.metric.start[[arc]],network.metric.vector[[arc]])
        fraction=float(fraction[0])
        if not reset:
            if arc==old_arc and (fraction-old_fraction)*network.lengths[arc]>=-params.backtrack_tolerance_m:
                addition=[]
            elif not bridge:
                addition=[arc]
            else:
                cutoff=min(params.max_route_distance_m,displacement+params.path_margin_m)
                middle,path=network.shortest_path(int(network.arcs[old_arc,1]),int(network.arcs[arc,0]),cutoff)
                total=(1-old_fraction)*network.lengths[old_arc]+middle+fraction*network.lengths[arc]
                if not np.isfinite(total) or total>cutoff:
                    reset=True
                else:
                    addition=[*path,arc]
        if reset:
            segments.append([arc])
        else:
            segments[-1].extend(addition)
        previous=(obs,arc,fraction)
    return segments
