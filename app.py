from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PP Virtual Quality Lab",
    page_icon="🧪",
    layout="wide"
)

# Resolve the model path relative to this script so it loads correctly
# regardless of the working directory Streamlit is launched from.
MODEL_PATH = Path(__file__).parent / "pp_virtual_lab_models .joblib"

TARGET_PROPERTIES = ["Izod Impact", "Tensile Modulus", "Flexural Modulus"]


@st.cache_resource(show_spinner="Loading prediction models...")
def load_model_artifact(path: Path) -> dict:
    """Load the trained model artifact and verify it has the expected shape.

    Expected structure:
        {
            "models": {target_name: fitted sklearn estimator, ...},
            "feature_columns": {target_name: [ordered feature names], ...},
            "metrics": {target_name: {"rows": int, "r2": float, "mae": float}, ...},
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"Model artifact has unexpected type {type(artifact).__name__}; "
            "expected a dict with 'models', 'feature_columns', and 'metrics'."
        )

    required_keys = {"models", "feature_columns", "metrics"}
    missing_keys = required_keys - artifact.keys()
    if missing_keys:
        raise ValueError(
            f"Model artifact is missing expected key(s): {sorted(missing_keys)}"
        )

    missing_targets = [
        target for target in TARGET_PROPERTIES
        if target not in artifact["models"]
        or target not in artifact["feature_columns"]
        or target not in artifact["metrics"]
    ]
    if missing_targets:
        raise ValueError(
            f"Model artifact is missing data for target(s): {missing_targets}"
        )

    return artifact


def build_feature_row(
    target: str,
    feature_columns: dict,
    grade: str,
    mfr: float,
    xs: float,
    c2: float,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the exact schema the target's
    model was trained on (raw numeric features + one-hot grade dummies).

    Raises ValueError if the selected grade was not part of the training
    schema for this target, instead of silently predicting on an
    all-zero / unknown grade encoding.
    """
    columns = feature_columns[target]
    row = {column: 0.0 for column in columns}

    for base_column, value in (("MFR", mfr), ("XS", xs), ("C2", c2)):
        if base_column not in row:
            raise ValueError(
                f"Expected feature '{base_column}' is missing from the "
                f"{target} model's trained schema."
            )
        row[base_column] = value

    grade_column = f"Grade_{grade}"
    if grade_column not in row:
        raise ValueError(
            f"Grade '{grade}' was not part of the training data for the "
            f"{target} model, so a reliable prediction cannot be produced "
            "for this grade."
        )
    row[grade_column] = 1.0

    return pd.DataFrame([row], columns=columns)


GRADE_OPTIONS = [
    "CA0900BM",
    "CB0900MO",
    "CB1248MO",
    "CB1640MO",
    "CB1849MO",
    "CB2000GT",
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
    artifact = load_model_artifact(MODEL_PATH)
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
        index=9
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
        predictions = {}
        for target in TARGET_PROPERTIES:
            feature_row = build_feature_row(
                target, feature_columns, grade, mfr, xs, c2
            )
            predictions[target] = models[target].predict(feature_row)[0]
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
