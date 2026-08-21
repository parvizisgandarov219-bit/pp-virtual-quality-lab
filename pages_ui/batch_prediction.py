"""Batch Prediction page."""

import streamlit as st

from batch_utils import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    parse_uploaded_file,
    summarize_batch_results,
    validate_and_predict_batch,
)
from model_utils import SUPPORTED_GRADES

_STATUS_STYLE = {
    "OK": "background-color: #E4F1E9; color: #2E7D4F;",
    "ERROR": "background-color: #F8E6E6; color: #B23B3B;",
}


def _highlight_status(row):
    style = _STATUS_STYLE.get(row.get("Validation Status"), "")
    return [style if col == "Validation Status" else "" for col in row.index]


def render(artifact: dict) -> None:
    models = artifact["models"]
    feature_columns = artifact["feature_columns"]

    st.markdown('<span class="pp-badge">Batch Prediction</span>', unsafe_allow_html=True)
    st.title("Predict a full batch")
    st.caption("Upload a CSV or Excel file to predict properties for many rows at once.")

    with st.expander("How it works", expanded=False):
        st.markdown(
            "Required columns: **Grade**, **MFR**, **XS**, **C2**.\n\n"
            "**Grade Family** is determined automatically from the Grade "
            "code - grades starting with **C** are HECO, **R** are RACO, "
            "**H** are HOMO. You don't need to supply it. If your file "
            "*does* include an optional **Grade Family** column, its value "
            "must agree with the grade (ICP is accepted as another name "
            "for HECO) - a row where they disagree is flagged as an error "
            "rather than silently corrected. HOMO rows may leave C2 blank "
            "and it defaults to 0; RACO and HECO rows always require it.\n\n"
            "Any other columns in your file - for example **Production Date** "
            "or **Batch Number** - are preserved unchanged in the results. "
            "Those two are treated as the batch's identity, useful later for "
            "matching predictions back to actual lab results."
        )

    uploaded_file = st.file_uploader(
        "Upload batch file",
        type=["csv", "xlsx"],
        help=f"Grade must be one of the {len(SUPPORTED_GRADES)} grades supported on the Dashboard.",
    )

    if uploaded_file is None:
        return

    if not st.button("Run Batch Predictions", type="primary"):
        return

    try:
        input_df = parse_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as exc:
        st.error(f"Could not read uploaded file: {exc}")
        return

    if input_df.empty:
        st.warning("The uploaded file has no data rows.")
        return

    try:
        result_df = validate_and_predict_batch(input_df, models, feature_columns)
    except Exception as exc:
        st.error(f"Could not process batch file: {exc}")
        return

    summary = summarize_batch_results(result_df)

    stat_cols = st.columns(3)
    with stat_cols[0]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n">{summary["total"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Total rows</div>', unsafe_allow_html=True)
    with stat_cols[1]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n ok">{summary["ok"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Succeeded</div>', unsafe_allow_html=True)
    with stat_cols[2]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n error">{summary["errors"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Failed validation</div>', unsafe_allow_html=True)

    if summary["errors"]:
        st.warning(
            f"{summary['errors']} row(s) failed validation - see the "
            "'Validation Status' / 'Validation Error' columns below. "
            "No prediction was computed for those rows."
        )

    try:
        styled = result_df.style.apply(_highlight_status, axis=1)
        st.dataframe(styled, width="stretch")
    except Exception:
        # Styling is a display nicety only - if it ever fails, still show
        # the (unstyled) real results rather than losing the data.
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
