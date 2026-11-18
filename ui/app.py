"""Browser-only demo client; API calls execute on the model host, never in JS."""
import os
import time
import folium
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="PATHSNAP · Route Lab", page_icon="🧭", layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem; max-width:1450px}
h1 {letter-spacing:-.045em} [data-testid="stMetric"] {background:#f0f5f3;padding:14px;border-radius:10px}
</style>""", unsafe_allow_html=True)
st.caption("PATHSNAP / ROUTE LAB")
st.title("From GPS noise to a connected route")
st.write("Run six saved model configurations on real trips. Compare independent arc choices with a directed HMM route.")


def call(method, path, **kwargs):
    response = requests.request(method, API_URL + path, timeout=kwargs.pop("timeout", 20), **kwargs)
    response.raise_for_status()
    return response.json()


try:
    health = call("GET", "/health")
    catalog = call("GET", "/models")
    trips = call("GET", "/trips")["trips"]
except (requests.RequestException, ValueError):
    st.error("The model host is unavailable. Start the API on the host laptop, then reload this page.")
    st.stop()
if not health.get("ready") or not trips:
    st.warning("The host is starting or has no student-testing trips. Check the host terminal.")
    st.stop()
models = {item["id"]: item for item in catalog["models"]}
st.session_state.setdefault("results", {})
st.session_state.setdefault("demo_queue", [])

with st.sidebar:
    st.header("Run a trip")
    trip = st.selectbox("Student-testing trip", trips)
    model = st.selectbox("Model", list(models), format_func=lambda value: models[value]["label"])
    show_hmm = st.checkbox("Show HMM route", value=True)
    busy = bool(st.session_state.get("active_job"))
    selected = st.button("Run selected model", type="primary", disabled=busy or not models[model]["available"], width="stretch")
    all_models = st.button("Demo every model", disabled=busy or not all(m["available"] for m in models.values()), width="stretch")
    st.caption("KNN can take up to a minute on a long trip. The job continues on the host while this page polls for progress.")
    st.divider()
    st.markdown("**Selected best**")
    st.write("Tuned + calibrated Histogram Boosting")
    st.caption(catalog["selection"])
    st.caption("Host connected · models and data stay on the host laptop")


def start_next():
    selected_trip, selected_model = st.session_state.demo_queue.pop(0)
    job = call("POST", "/jobs", json=dict(trip_id=selected_trip, model_name=selected_model))
    st.session_state.active_job = dict(id=job["job_id"], trip=selected_trip, model=selected_model, started=time.monotonic())


if selected or all_models:
    order = ["knn", "mlp", "logistic_regression", "random_forest", "hist_gradient_boosting", "best"] if all_models else [model]
    st.session_state.demo_queue = [(trip, name) for name in order]
    st.session_state.pop("job_error", None)
    try:
        start_next()
        st.rerun()
    except requests.RequestException as exc:
        st.session_state.demo_queue = []
        st.error(f"Could not start prediction: {exc}")


@st.fragment(run_every="1s")
def progress():
    job = st.session_state.get("active_job")
    if not job:
        return
    elapsed = time.monotonic() - job["started"]
    st.info(f"Running {models[job['model']]['label']} · trip {job['trip']} · {elapsed:.0f}s elapsed · {len(st.session_state.demo_queue)} remaining")
    try:
        status = call("GET", f"/jobs/{job['id']}")["status"]
        if status == "complete":
            result = call("GET", f"/jobs/{job['id']}/result", timeout=60)
            st.session_state.results[(job["trip"], job["model"])] = result
            del st.session_state.active_job
            if st.session_state.demo_queue:
                start_next()
            st.rerun()
        elif status == "failed":
            st.session_state.job_error = "Prediction failed. Check the host API log, then run this model again."
            st.session_state.demo_queue = []
            del st.session_state.active_job
            st.rerun()
    except requests.RequestException:
        st.warning("Connection interrupted. Retrying the existing job; inference is not restarted.")


progress()
if st.session_state.get("job_error"):
    st.error(st.session_state.job_error)

with st.expander("Why this model? · recorded comparison", expanded=False):
    st.caption("Historical classifier Top-1 on approximate known point labels. CV selected the model; student test is for reporting. Final calibration preserves the tuned model's arc ranking.")
    st.dataframe(pd.DataFrame([dict(Model=m["label"], **{"Training CV Top-1":m["cv_top1"], "Student Top-1":m["student_top1"], "Selected":m["selected"]}) for m in models.values()]),
                 hide_index=True, column_config={"Training CV Top-1":st.column_config.NumberColumn(format="percent"), "Student Top-1":st.column_config.NumberColumn(format="percent")}, width="stretch")
    st.caption(catalog["decoder"])

history = [value for (tid, _), value in st.session_state.results.items() if tid == trip]
if history:
    st.subheader("This trip · live model comparison")
    rows = []
    for value in history:
        metrics = value.get("metrics")
        rows.append(dict(Model=models[value["model"]]["label"], **{
            "Classifier accuracy":metrics["classifier"]["pointwise_accuracy"] if metrics else None,
            "HMM accuracy":metrics["viterbi"]["pointwise_accuracy"] if metrics else None,
            "HMM sequence F1":metrics["viterbi"]["sequence_f1"] if metrics else None,
            "Server seconds":value["timing_ms"]/1000}))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={key:st.column_config.NumberColumn(format="%.3f") for key in ("Classifier accuracy", "HMM accuracy", "HMM sequence F1", "Server seconds")})

result = st.session_state.results.get((trip, model))
if result is None:
    st.info("Choose a model and run a prediction. “Demo every model” runs the five families and finishes with the selected best model.")
    st.stop()
st.subheader(f"Trip {trip} · {models[model]['label']}")
a, b, c, d = st.columns(4)
a.metric("GPS observations", result["n_observations"])
b.metric("Server latency", f"{result['timing_ms']/1000:.2f} s")
metrics = result.get("metrics")
if metrics:
    cls, hmm = metrics["classifier"], metrics["viterbi"]
    c.metric("Classifier accuracy", f"{cls['pointwise_accuracy']:.1%}" if cls["pointwise_accuracy"] is not None else "—")
    d.metric("HMM accuracy", f"{hmm['pointwise_accuracy']:.1%}" if hmm["pointwise_accuracy"] is not None else "—")
    st.caption(f"Approximate aligned point labels · {metrics['n_known_obs']:,} known / {metrics['n_unknown_obs']:,} unknown · reference geometry is never sent to this dashboard.")
    st.dataframe(pd.DataFrame([dict(Method=name, **values) for name, values in (("Classifier", cls), ("HMM + directed bridges", hmm))]), hide_index=True, width="stretch")
else:
    st.warning("Reference evaluation unavailable. The map still shows live predictions.")

# Every road arc is a separate line: disconnected predictions are never joined.
gps = result["gps_geometry"]
route_map = folium.Map(location=gps[len(gps)//2], tiles="OpenStreetMap", prefer_canvas=True)

def layer(name, segments, color, weight, show=True, opacity=.85):
    group = folium.FeatureGroup(name=name, show=show)
    coordinates = [[[lon, lat] for lat, lon in segment] for segment in segments]
    folium.GeoJson({"type":"Feature", "properties":{}, "geometry":{"type":"MultiLineString", "coordinates":coordinates}},
                   style_function=lambda _: dict(color=color, weight=weight, opacity=opacity)).add_to(group)
    group.add_to(route_map)

layer("Road network", result["road_geometry"], "#87938e", 1, show=False, opacity=.4)
layer("Classifier arc choices", result["classifier_geometry"], "#2274ce", 3)
layer("HMM + directed bridges", result["predicted_geometry"], "#8a31ae", 4, show=show_hmm)
group = folium.FeatureGroup(name="GPS observations")
for index in range(0, len(gps), max(1, (len(gps)+699)//700)):
    folium.CircleMarker(gps[index], radius=2, color="#e46e32", fill=True,
                        tooltip=f"Point {index} · t={result['timestamps'][index]:g}s · classifier {result['classifier_arcs'][index]} · HMM {result['predicted_arcs'][index]}").add_to(group)
group.add_to(route_map)
folium.LayerControl(collapsed=False).add_to(route_map)
route_map.fit_bounds(gps)
st.caption("Orange: GPS · Blue: classifier · Purple: HMM with directed bridges. Toggle layers on the map.")
st_folium(route_map, height=580, use_container_width=True, returned_objects=[], key=f"map-{trip}-{model}-{show_hmm}")
with st.expander("Timing and decoder diagnostics"):
    st.json(dict(stages_ms=result["stages_ms"], diagnostics=result["diagnostics"]))
st.download_button("Download this prediction JSON", data=__import__("json").dumps(result, indent=2), file_name=f"pathsnap_{trip}_{model}.json", mime="application/json")
