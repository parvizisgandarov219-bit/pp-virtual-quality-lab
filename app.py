from pathlib import Path

import streamlit as st

from model_utils import (
    C2_BOUNDS,
    MFR_BOUNDS,
    SUPPORTED_GRADES,
    XS_BOUNDS,
    load_model_artifact,
    log_prediction,
    predict_all,
)
from batch_utils import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    parse_uploaded_file,
    summarize_batch_results,
    validate_and_predict_batch,
)

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


GRADE_OPTIONS = SUPPORTED_GRADES

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
        min_value=MFR_BOUNDS[0],
        max_value=MFR_BOUNDS[1],
        value=48.0,
        step=0.1,
        help=f"Typical PP melt flow rate range: {MFR_BOUNDS[0]:g}-{MFR_BOUNDS[1]:g} g/10 min."
    )

with middle2:
    xs = st.number_input(
        "XS, wt%",
        min_value=XS_BOUNDS[0],
        max_value=XS_BOUNDS[1],
        value=16.0,
        step=0.1,
        help=f"Xylene solubles, typical range: {XS_BOUNDS[0]:g}-{XS_BOUNDS[1]:g} wt%."
    )

with right:
    c2 = st.number_input(
        "C2, wt%",
        min_value=C2_BOUNDS[0],
        max_value=C2_BOUNDS[1],
        value=8.6,
        step=0.1,
        help=f"Ethylene comonomer content, typical range: {C2_BOUNDS[0]:g}-{C2_BOUNDS[1]:g} wt%."
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

st.divider()

st.subheader("Batch Prediction")

st.caption(
    "Upload a CSV or Excel file with columns **Grade**, **MFR**, **XS**, "
    "and **C2**. An optional **Grade Family** column (HOMO / RACO / HECO / "
    "ICP) may be included - if present, HOMO rows may leave C2 blank "
    "(it defaults to 0). Without a Grade Family column, C2 is required "
    "for every row, same as the single prediction above."
)

uploaded_file = st.file_uploader(
    "Upload batch file",
    type=["csv", "xlsx"],
    help="Grade must be one of the 11 grades supported above."
)

if uploaded_file is not None:
    if st.button("Run Batch Predictions", type="primary"):
        try:
            input_df = parse_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
        else:
            if input_df.empty:
                st.warning("The uploaded file has no data rows.")
            else:
                try:
                    result_df = validate_and_predict_batch(input_df, models, feature_columns)
                except Exception as exc:
                    st.error(f"Could not process batch file: {exc}")
                else:
                    summary = summarize_batch_results(result_df)
                    st.success(
                        f"Processed {summary['total']} row(s): "
                        f"{summary['ok']} succeeded, {summary['errors']} failed validation."
                    )
                    if summary["errors"]:
                        st.warning(
                            f"{summary['errors']} row(s) failed validation - see the "
                            "'Validation Status' / 'Validation Error' columns below. "
                            "No prediction was computed for those rows."
                        )

                    st.dataframe(result_df, width="stretch")

                    download_col1, download_col2 = st.columns(2)
                    download_col1.download_button(
                        "Download results (CSV)",
                        data=dataframe_to_csv_bytes(result_df),
                        file_name="pp_batch_predictions.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                    download_col2.download_button(
                        "Download results (Excel)",
                        data=dataframe_to_excel_bytes(result_df),
                        file_name="pp_batch_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                    )
