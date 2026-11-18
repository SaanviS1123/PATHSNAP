"""Directed road graph with bounded, cached Dijkstra searches and arc paths."""
from collections import OrderedDict
import heapq
import math
import numpy as np
from src.data_processing.geo import project_trip
from src.data_processing.loaders import load_trip


class RoadNetwork:
    def __init__(self,metric,cache_size=2048):
        self.metric=metric
        self.arcs=metric.trip.arcs
        self.lengths=metric.length
        self.xy=metric.xy
        self.timestamps=metric.trip.track[:,2]
        self.trip_id=metric.trip.trip_id
        self.order=np.argsort(self.arcs[:,0],kind='stable')
        self.offsets=np.r_[0,np.cumsum(np.bincount(self.arcs[:,0],minlength=len(metric.node_xy)))]
        self.cache_size=cache_size
        self.cache=OrderedDict()
        self.searches=0
        self.hits=0
        self.visited_nodes=0

    @classmethod
    def from_trip_dir(cls,path):
        return cls(project_trip(load_trip(path)))  # Never reads .route.

    def successors(self,arc_id):
        node=int(self.arcs[arc_id,1])
        return self.order[self.offsets[node]:self.offsets[node+1]]

    def bounded_search(self,source,cutoff):
        source=int(source)
        if cutoff<0 or not np.isfinite(cutoff):
            raise ValueError('Cutoff must be nonnegative and finite')
        bucket=int(math.ceil(cutoff/100)*100)
        key=(source,bucket)
        if key in self.cache:
            self.hits+=1
            self.cache.move_to_end(key)
            return self.cache[key]
        distances={source:0.}
        parents={}
        queue=[(0.,source)]
        while queue:
            cost,node=heapq.heappop(queue)
            if cost!=distances[node]:
                continue
            for arc in self.order[self.offsets[node]:self.offsets[node+1]]:
                target=int(self.arcs[arc,1])
                proposed=cost+float(self.lengths[arc])
                if proposed<=bucket and proposed<distances.get(target,np.inf):
                    distances[target]=proposed
                    parents[target]=int(arc)
                    heapq.heappush(queue,(proposed,target))
        self.searches+=1
        self.visited_nodes+=len(distances)
        self.cache[key]=(distances,parents)
        if len(self.cache)>self.cache_size:
            self.cache.popitem(last=False)
        return distances,parents

    def shortest_path(self,source,target,cutoff):
        distances,parents=self.bounded_search(source,cutoff)
        target=int(target)
        distance=distances.get(target,np.inf)
        if distance>cutoff:
            return np.inf,[]
        path=[]
        node=target
        while node!=source:
            arc=parents[node]
            path.append(arc)
            node=int(self.arcs[arc,0])
        return float(distance),path[::-1]

    def transition_distances(self,previous,current,previous_fraction,current_fraction,cutoff,backtrack_m=20.):
        remaining=(1-previous_fraction)*self.lengths[previous]
        along=current_fraction*self.lengths[current]
        ends=self.arcs[previous,1]
        starts=self.arcs[current,0]
        middle=np.empty((len(previous),len(current)))
        for source in np.unique(ends):
            distances,_=self.bounded_search(int(source),cutoff)
            row=np.array([distances.get(int(target),np.inf) for target in starts])
            middle[ends==source]=row
        total=remaining[:,None]+middle+along[None,:]
        for i,arc in enumerate(previous):
            same=np.flatnonzero(current==arc)
            if len(same):
                delta=(current_fraction[same]-previous_fraction[i])*self.lengths[arc]
                allowed=delta>=-backtrack_m
                total[i,same[allowed]]=np.maximum(0,delta[allowed])
        total[total>cutoff]=np.inf
        return total

    def stats(self):
        return dict(shortest_path_searches=self.searches,cache_hits=self.hits,
                    cached_sources=len(self.cache),visited_nodes=self.visited_nodes)
