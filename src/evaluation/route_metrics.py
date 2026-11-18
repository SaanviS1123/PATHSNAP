"""C1: comparable pointwise/direct continuity plus ordered route metrics."""
from bisect import bisect_left
from collections import defaultdict
import numpy as np
from src.evaluation.continuity import route_continuity as _direct_continuity


def route_continuity(predictions,network=None,networks=None):
    if network is not None:
        networks={network.trip_id:network.arcs}
    if networks is not None:
        networks={str(key):(value.arcs if hasattr(value,'arcs') else value) for key,value in networks.items()}
    # Missing states break adjacent pairs; they are never used as negative indexes.
    if (predictions.arc_id<0).any():
        valid=predictions[predictions.arc_id>=0]
        good=0;total=0
        for trip_id,group in predictions.groupby('trip_id'):
            total+=max(0,len(group)-1)
            present=valid[valid.trip_id==trip_id]
            if len(present)>1:
                ratio=_direct_continuity(present,networks)
                good+=round(ratio*(len(present)-1))
        return float(good/total) if total else None
    return _direct_continuity(predictions,networks)


def pointwise_accuracy(predictions,labels):
    if labels.duplicated(['trip_id','obs_idx']).any() or predictions.duplicated(['trip_id','obs_idx']).any():
        raise ValueError('Duplicate observation keys')
    known=labels[labels.ref_arc>=0]
    joined=known[['trip_id','obs_idx','ref_arc']].merge(predictions[['trip_id','obs_idx','arc_id']],on=['trip_id','obs_idx'],how='left',validate='one_to_one')
    correct=joined.arc_id==joined.ref_arc
    return dict(pointwise_accuracy=float(correct.mean()) if len(known) else None,n_known_obs=len(known),n_unknown_obs=int((labels.ref_arc<0).sum()),n_correct=int(correct.sum()))


def lcs_length(predicted,reference):
    # Hunt–Szymanski: repeated arc occurrences remain distinct ordered matches.
    positions=defaultdict(list)
    for index,arc in enumerate(reference):
        positions[int(arc)].append(index)
    tails=[]
    for arc in predicted:
        for position in reversed(positions.get(int(arc),[])):
            index=bisect_left(tails,position)
            if index==len(tails):
                tails.append(position)
            else:
                tails[index]=position
    return len(tails)


def sequence_metrics(segments,reference):
    predicted=[int(arc) for segment in segments for arc in segment]
    reference=list(map(int,reference))
    overlap=lcs_length(predicted,reference)
    precision=overlap/len(predicted) if predicted else float(not reference)
    recall=overlap/len(reference) if reference else float(not predicted)
    return dict(sequence_precision=precision,sequence_recall=recall,
                sequence_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                lcs_arcs=overlap,predicted_route_arcs=len(predicted),reference_route_arcs=len(reference),
                exact_route_match=predicted==reference,n_route_segments=len(segments))
