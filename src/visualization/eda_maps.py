"""Partition-labelled static EDA and a train-only Folium GPS/network map."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pathsnap-matplotlib')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import folium
import numpy as np
from src.data_processing.loaders import load_trip, trip_directories


def create_eda(quality, raw_dir):
    out = Path('results/plots')
    out.mkdir(parents=True, exist_ok=True)
    for column, title, xlabel in [('n_obs', 'GPS observations per trip', 'Observations'),
                                 ('n_arcs', 'Road-network sizes', 'Directed arcs'),
                                 ('max_gap_s', 'Maximum timestamp gap per trip', 'Seconds'),
                                 ('missing_values', 'Missing GPS values per trip', 'Missing values')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for partition, group in quality.groupby('partition'):
            ax.hist(group[column], bins=20, alpha=.55, label=partition)
        ax.set(title=title, xlabel=xlabel, ylabel='Trips')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'eda_{column}.png', dpi=150)
        plt.close(fig)
    trip = load_trip(trip_directories(raw_dir, 'train')[0])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(LineCollection(trip.nodes[trip.arcs], linewidths=.25, colors='grey', alpha=.3))
    ax.plot(*trip.track[:, :2].T, color='#c24b22', linewidth=1, label='GPS')
    ax.autoscale()
    ax.set(title=f'Train trip {trip.trip_id}: GPS and road network', xlabel='Longitude (degrees)', ylabel='Latitude (degrees)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'eda_trajectory.png', dpi=150)
    plt.close(fig)
    centre = trip.track[:, :2].mean(axis=0)[::-1]
    map_ = folium.Map(location=centre, tiles='OpenStreetMap', zoom_start=12)
    road = folium.FeatureGroup(name=f'Train {trip.trip_id}: road network')
    # GeoJSON avoids tens of thousands of individually scripted Leaflet layers.
    geometry = {'type': 'Feature', 'properties': {}, 'geometry': {'type': 'MultiLineString', 'coordinates': trip.nodes[trip.arcs].tolist()}}
    folium.GeoJson(geometry, style_function=lambda _: {'color': '#657785', 'weight': 1, 'opacity': .45}).add_to(road)
    road.add_to(map_)
    gps = folium.FeatureGroup(name='Raw GPS observations')
    folium.PolyLine(trip.track[:, :2][:, ::-1].tolist(), color='#d65128', weight=3).add_to(gps)
    for lon, lat, timestamp in trip.track:
        folium.CircleMarker([lat, lon], radius=2, tooltip=f't={timestamp:g}s', color='#d65128').add_to(gps)
    gps.add_to(map_)
    folium.LayerControl().add_to(map_)
    map_.fit_bounds([trip.track[:, :2].min(axis=0)[::-1].tolist(), trip.track[:, :2].max(axis=0)[::-1].tolist()])
    map_.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:12px">Train EDA · orange: GPS · grey: road network</div>'))
    Path('maps').mkdir(exist_ok=True)
    map_.save(f'maps/eda_train_{trip.trip_id}.html')
