
"""PATHSNAP Streamlit frontend."""

import os

import requests
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PATHSNAP",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ PATHSNAP")
st.subheader("Distributed Map-Matching & Route Reconstruction")

st.markdown(
    """
    Select a student-testing trip and request a route prediction
    from the PATHSNAP API.
    """
)

# --------------------------------------------------
# API configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 30


# --------------------------------------------------
# API helper functions
# --------------------------------------------------

def check_api_health():
    """Check whether the backend is available."""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json(), None

    except requests.RequestException as exc:
        return None, str(exc)


def fetch_trips():
    """Retrieve permitted student-testing trip IDs."""
    try:
        response = requests.get(
            f"{API_URL}/trips",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()
        return data.get("trips", []), None

    except requests.RequestException as exc:
        return [], str(exc)


def request_prediction(trip_id):
    """Send a selected trip to the prediction API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"trip_id": trip_id},
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 503:
            return None, (
                "The prediction pipeline has not been "
                "connected yet. No prediction was generated."
            )

        response.raise_for_status()
        return response.json(), None

    except requests.RequestException as exc:
        return None, str(exc)


# --------------------------------------------------
# Sidebar: backend status
# --------------------------------------------------

st.sidebar.header("Backend Status")

health_data, health_error = check_api_health()

if health_data:
    st.sidebar.success("API connected")
    st.sidebar.caption(
        f"Project: {health_data.get('project', 'PATHSNAP')}"
    )
    st.sidebar.caption(
        f"Version: {health_data.get('version', 'Unknown')}"
    )
else:
    st.sidebar.error("API unavailable")
    st.sidebar.caption(health_error)

st.sidebar.markdown("---")
st.sidebar.caption(f"API URL: {API_URL}")


# --------------------------------------------------
# Main: trip selection
# --------------------------------------------------

st.header("Trip Selection")

trips, trips_error = fetch_trips()

if trips_error:
    st.error(f"Could not load trips: {trips_error}")
    st.info(
        "Make sure the FastAPI backend is running "
        "and the API URL is correct."
    )
    st.stop()

if not trips:
    st.warning("No student-testing trips were found.")
    st.stop()

st.success(f"{len(trips)} student-testing trips available.")

selected_trip = st.selectbox(
    "Choose a trip ID",
    options=trips,
)

st.write(f"**Selected trip:** `{selected_trip}`")


# --------------------------------------------------
# Prediction request
# --------------------------------------------------

st.header("Route Prediction")

if st.button(
    "Predict Route",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Sending trip to PATHSNAP API..."):
        prediction, prediction_error = request_prediction(
            selected_trip
        )

    if prediction_error:
        st.warning(prediction_error)

    elif prediction:
        st.session_state["prediction"] = prediction
        st.session_state["prediction_trip"] = selected_trip
        st.success("Prediction received successfully!")


# --------------------------------------------------
# Display prediction response
# --------------------------------------------------

if "prediction" in st.session_state:
    prediction = st.session_state["prediction"]

    if st.session_state.get("prediction_trip") == selected_trip:
        st.subheader("Prediction Result")

        st.json(prediction)

        if prediction.get("predicted_geometry"):
            st.info(
                "Route geometry received. Map visualisation "
                "will be added when the pipeline is integrated."
            )
        else:
            st.caption(
                "No route geometry is available yet."
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.caption(
    "PATHSNAP | Distributed Map-Matching & Route Reconstruction"
)

