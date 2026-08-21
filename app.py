from pathlib import Path

import streamlit as st

from model_utils import load_model_artifact, log_prediction, predict_all

st.set_page_config(
    page_title="PP Virtual Quality Lab",
    page_icon="🧪",
    layout="wide"
)

# Resolve paths relative to this script so they work correctly regardless
# of the working directory Streamlit is launched from.
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "pp_virtual_lab_models .joblib"

# Best-effort local audit log of predictions made. NOTE: on Streamlit
# Community Cloud the filesystem is ephemeral, so this file will not
# survive an app restart/redeploy there - see README for details.
PREDICTION_LOG_PATH = APP_DIR / "predictions_log.csv"


@st.cache_resource(show_spinner="Loading prediction models...")
def _load_cached_model_artifact(path: Path) -> dict:
    return load_model_artifact(path)


GRADE_OPTIONS = [
    "CA0900BM",
    "CB0900MO",
    "CB1248MO",
    "CB1640MO",
    "CB1849MO",
    "CB3000GT",
    "CB3648MO",
    "CB4048MO",
    "CB4848MO",
    "CB6448MO",
    "CB8248MO"
]

st.title("PP Virtual Quality Lab")

st.info(
    "Decision-support model only. "
    "Laboratory confirmation remains required."
)

try:
    artifact = _load_cached_model_artifact(MODEL_PATH)
    load_error = None
except Exception as exc:
    artifact = None
    load_error = str(exc)

if load_error:
    st.error(
        "Prediction models could not be loaded, so this tool is "
        f"currently unavailable. Details: {load_error}"
    )
    st.stop()

models = artifact["models"]
feature_columns = artifact["feature_columns"]
metrics = artifact["metrics"]

left, middle1, middle2, right = st.columns(4)

with left:
    grade = st.selectbox(
        "Grade",
        GRADE_OPTIONS,
        index=8
    )

with middle1:
    mfr = st.number_input(
        "MFR, g/10 min",
        min_value=0.0,
        max_value=150.0,
        value=48.0,
        step=0.1,
        help="Typical PP melt flow rate range: 0-150 g/10 min."
    )

with middle2:
    xs = st.number_input(
        "XS, wt%",
        min_value=0.0,
        max_value=40.0,
        value=16.0,
        step=0.1,
        help="Xylene solubles, typical range: 0-40 wt%."
    )

with right:
    c2 = st.number_input(
        "C2, wt%",
        min_value=0.0,
        max_value=25.0,
        value=8.6,
        step=0.1,
        help="Ethylene comonomer content, typical range: 0-25 wt%."
    )

if st.button(
    "Calculate Properties",
    type="primary",
    use_container_width=True
):
    try:
        predictions = predict_all(models, feature_columns, grade, mfr, xs, c2)
    except Exception as exc:
        st.error(f"Could not compute predictions: {exc}")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Izod Impact",
            f"{predictions['Izod Impact']:.2f}"
        )

        col2.metric(
            "Predicted Tensile Modulus",
            f"{predictions['Tensile Modulus']:.0f} MPa"
        )

        col3.metric(
            "Predicted Flexural Modulus",
            f"{predictions['Flexural Modulus']:.0f} MPa"
        )

        try:
            log_prediction(PREDICTION_LOG_PATH, grade, mfr, xs, c2, predictions)
        except Exception:
            # Logging is best-effort only and must never break the UI.
            pass

st.divider()

st.subheader("Model Validation")

v1, v2, v3 = st.columns(3)

v1.metric("Izod Impact R²", f"{metrics['Izod Impact']['r2']:.3f}")
v2.metric("Tensile Modulus R²", f"{metrics['Tensile Modulus']['r2']:.3f}")
v3.metric("Flexural Modulus R²", f"{metrics['Flexural Modulus']['r2']:.3f}")

st.warning(
    "Impact is the strongest validated prediction. "
    "Tensile and Flexural are early estimates and "
    "must not be used alone for product release."
)
