
"""PATHSNAP Streamlit interface."""

import os

import folium
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium


load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="PATHSNAP", page_icon="🗺️", layout="wide")

st.title("🗺️ PATHSNAP")
st.subheader("Distributed Map-Matching and Route Reconstruction")
st.write("Compare classifier-only predictions with classifier + Viterbi.")


# API connection
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    st.success("PATHSNAP API connected")
except requests.RequestException as exc:
    st.error(f"Cannot connect to API: {exc}")
    st.stop()


# Available trips
try:
    response = requests.get(f"{API_URL}/trips", timeout=10)
    response.raise_for_status()
    trip_ids = response.json().get("trips", [])
except requests.RequestException as exc:
    st.error(f"Could not load trips: {exc}")
    st.stop()

if not trip_ids:
    st.warning("No student-testing trips found.")
    st.stop()


selected_trip = st.selectbox("Select student-testing trip", trip_ids)

if st.button("Run route prediction", type="primary", use_container_width=True):
    with st.spinner("Running prediction and server-side evaluation..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"trip_id": selected_trip},
                timeout=300,
            )
            response.raise_for_status()
            st.session_state["prediction_result"] = response.json()
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")
            if getattr(exc, "response", None) is not None:
                st.code(exc.response.text)
            st.stop()


result = st.session_state.get("prediction_result")

if result is None or result.get("trip_id") != selected_trip:
    st.info("Select a trip and click Run route prediction.")
    st.stop()

if result.get("status") != "success":
    st.error(result.get("message", "Prediction failed."))
    st.stop()

st.success(result.get("message", "Prediction completed."))

# Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("GPS observations", result.get("n_observations", 0))
c2.metric("Candidate road segments", result.get("n_road_segments", 0))
c3.metric("Classifier predictions", len(result.get("classifier_arcs", [])))
c4.metric("Viterbi predictions", len(result.get("predicted_arcs", [])))

st.caption(f"Prediction time: {result.get('timing_ms', 0):.2f} ms")


# Evaluation metrics
st.markdown("## Evaluation results")

metrics = result.get("metrics")

if metrics:
    classifier = metrics["classifier"]
    viterbi = metrics["viterbi"]

    st.caption(
        "Reference routes are evaluated on the API server. "
        "Only aggregate numerical metrics are sent to this interface."
    )

    metric_names = [
        ("Sequence precision", "sequence_precision"),
        ("Sequence recall", "sequence_recall"),
        ("Sequence F1", "sequence_f1"),
        ("Route continuity", "route_continuity"),
    ]

    for label, key in metric_names:
        left, right = st.columns(2)
        left.metric(
            f"{label} — Classifier",
            f"{classifier[key] * 100:.2f}%",
        )
        right.metric(
            f"{label} — Viterbi",
            f"{viterbi[key] * 100:.2f}%",
            delta=f"{(viterbi[key] - classifier[key]) * 100:+.2f} pp",
        )
else:
    st.warning("Reference evaluation metrics are unavailable for this trip.")


# Interactive Folium map
gps = result.get("gps_geometry", [])
roads = result.get("road_geometry", [])
classifier_route = result.get("classifier_geometry", [])
viterbi_route = result.get("predicted_geometry", [])

if not gps:
    st.warning("No GPS geometry was returned.")
    st.stop()

center = [
    sum(point[0] for point in gps) / len(gps),
    sum(point[1] for point in gps) / len(gps),
]

route_map = folium.Map(
    location=center,
    zoom_start=13,
    control_scale=True,
)

candidate_layer = folium.FeatureGroup(
    name="Candidate road network",
    show=True,
)
for segment in roads:
    folium.PolyLine(
        segment, color="gray", weight=1, opacity=0.45
    ).add_to(candidate_layer)
candidate_layer.add_to(route_map)

gps_layer = folium.FeatureGroup(
    name="Original GPS observations",
    show=True,
)
folium.PolyLine(
    gps, color="blue", weight=3, opacity=0.8
).add_to(gps_layer)

for index, point in enumerate(gps):
    if index in (0, len(gps) - 1):
        folium.CircleMarker(
            point,
            radius=5,
            color="blue",
            fill=True,
            tooltip="Start" if index == 0 else "End",
        ).add_to(gps_layer)

gps_layer.add_to(route_map)

classifier_layer = folium.FeatureGroup(
    name="Classifier-only route",
    show=True,
)
for segment in classifier_route:
    folium.PolyLine(
        segment, color="orange", weight=3, opacity=0.8
    ).add_to(classifier_layer)
classifier_layer.add_to(route_map)

viterbi_layer = folium.FeatureGroup(
    name="Classifier + Viterbi route",
    show=True,
)
for segment in viterbi_route:
    folium.PolyLine(
        segment, color="red", weight=4, opacity=0.9
    ).add_to(viterbi_layer)
viterbi_layer.add_to(route_map)

folium.LayerControl(collapsed=False).add_to(route_map)

st.markdown("## Interactive route comparison")
st_folium(
    route_map,
    use_container_width=True,
    height=650,
    returned_objects=[],
)

st.caption(
    "Reference route IDs are never returned to or displayed in the UI."
)
