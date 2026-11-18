"""Prediction maps; references are explicit offline-only inputs, never auto-loaded."""
from pathlib import Path
import html
import folium
import numpy as np

COLORS={'gps':'#d45a25','network':'#7c8991','reference':'#278845','classifier':'#176bca','hmm':'#8a31ae'}


def prediction_map(result,path,reference_route=None):
    network=result.network
    trip=network.metric.trip
    coordinates=trip.track[:,:2]
    map_=folium.Map(location=coordinates.mean(axis=0)[::-1].tolist(),tiles='OpenStreetMap',zoom_start=13)
    lower=network.xy.min(axis=0)-200
    upper=network.xy.max(axis=0)+200
    endpoints=network.metric.node_xy[network.arcs]
    near=np.flatnonzero((endpoints.max(axis=1)>=lower).all(axis=1)&(endpoints.min(axis=1)<=upper).all(axis=1))
    def layer(name,arc_ids,color,weight,show=True,opacity=.9,dash=None):
        ids=np.asarray(arc_ids,dtype=int)
        ids=np.unique(ids[ids>=0])
        geometry={'type':'Feature','properties':{},'geometry':{'type':'MultiLineString','coordinates':trip.nodes[trip.arcs[ids]].tolist()}}
        group=folium.FeatureGroup(name=name,show=show)
        style={'color':color,'weight':weight,'opacity':opacity}
        if dash:
            style['dashArray']=dash
        folium.GeoJson(geometry,style_function=lambda _,style=style:style).add_to(group)
        group.add_to(map_)
    layer('Road network (cropped)',near,COLORS['network'],1,show=False,opacity=.45)
    if reference_route is not None:
        layer('Reference route (offline report only)',reference_route,COLORS['reference'],8,opacity=.5)
    layer('Classifier-only arc choices',result.classifier.arc_id.to_numpy(),COLORS['classifier'],3,dash='5 5')
    expanded=[arc for segment in result.segments for arc in segment]
    layer('HMM route with directed bridges',expanded,COLORS['hmm'],4)
    gps=folium.FeatureGroup(name='Raw GPS observations')
    for i in range(0,len(coordinates),max(1,len(coordinates)//700)):
        folium.CircleMarker(coordinates[i,::-1].tolist(),radius=2,color=COLORS['gps'],fill=True,
                            tooltip=f'obs {i}; t={trip.track[i,2]:g}s').add_to(gps)
    gps.add_to(map_)
    folium.LayerControl(collapsed=False).add_to(map_)
    map_.fit_bounds([coordinates.min(axis=0)[::-1].tolist(),coordinates.max(axis=0)[::-1].tolist()])
    labels=[('gps','GPS'),('classifier','classifier'),('hmm','HMM + bridges')]
    if reference_route is not None:
        labels.append(('reference','offline reference'))
    legend=' · '.join(f'<span style="color:{COLORS[key]}">{label}</span>' for key,label in labels)
    map_.get_root().html.add_child(folium.Element(f'<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:12px;font:13px sans-serif">Trip {html.escape(trip.trip_id)} · {legend}</div>'))
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    map_.save(path)
    return path
