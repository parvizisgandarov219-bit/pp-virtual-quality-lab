"""Model Validation page.

Lets an engineer upload a NEW lab results file - actual measured
properties for batches already produced - and checks the EXISTING model's
predictions against those real measurements. This is comparison/scoring
only: it never retrains, refits, or otherwise changes the deployed model.
The intent is to periodically re-run this as new lab data accumulates, to
see whether the model's real-world accuracy has drifted from its
original training-time metrics.
"""

import streamlit as st

from batch_utils import dataframe_to_csv_bytes, dataframe_to_excel_bytes, parse_uploaded_file
from model_utils import TARGET_PROPERTIES
from validation_utils import (
    compute_validation_metrics,
    summarize_validation_results,
    validate_and_score_batch,
)

_STATUS_STYLE = {
    "OK": "background-color: #E4F1E9; color: #2E7D4F;",
    "ERROR": "background-color: #F8E6E6; color: #B23B3B;",
}


def _highlight_status(row):
    style = _STATUS_STYLE.get(row.get("Validation Status"), "")
    return [style if col == "Validation Status" else "" for col in row.index]


def _fmt(value: float | None, digits: int) -> str:
    return f"{value:.{digits}f}" if value is not None else "—"


def render(artifact: dict) -> None:
    models = artifact["models"]
    feature_columns = artifact["feature_columns"]
    training_metrics = artifact["metrics"]

    st.markdown('<span class="pp-badge">Model Validation</span>', unsafe_allow_html=True)
    st.title("Validate against new lab data")
    st.caption(
        "Upload a file of actual lab measurements to check whether the "
        "EXISTING model's predictions still match real results. This "
        "never retrains or changes the model - it only scores it."
    )

    with st.expander("How it works", expanded=False):
        st.markdown(
            "Required columns: **Grade**, **MFR**, **XS**, **C2**, "
            "**Actual Izod Impact**, **Actual Tensile Modulus**, "
            "**Actual Flexural Modulus**.\n\n"
            "**Production Date** and **Batch Number**, if present, are "
            "metadata only - preserved in the results, never read by the "
            "model.\n\n"
            "Each row is validated and predicted exactly like Batch "
            "Prediction (same Grade/Family/C2 rules, any of the model's "
            "trained grades accepted). Rows that fail validation, or have "
            "a missing/non-numeric Actual value, are excluded from the "
            "R² / MAE / RMSE calculation below - one bad row never blocks "
            "the rest of the file.\n\n"
            "**R², MAE, and RMSE are computed only from successfully "
            "scored rows**, comparing Predicted vs. Actual for each "
            "property independently. This is model *validation*, not "
            "retraining - the deployed model is never changed by this "
            "page."
        )

    uploaded_file = st.file_uploader(
        "Upload lab results file",
        type=["csv", "xlsx"],
        help=(
            "Must include Grade, MFR, XS, C2, and Actual Izod Impact / "
            "Actual Tensile Modulus / Actual Flexural Modulus columns."
        ),
    )

    if uploaded_file is None:
        return

    if not st.button("Run Validation", type="primary"):
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
        result_df = validate_and_score_batch(input_df, models, feature_columns)
    except Exception as exc:
        st.error(f"Could not process validation file: {exc}")
        return

    summary = summarize_validation_results(result_df)

    stat_cols = st.columns(3)
    with stat_cols[0]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n">{summary["total"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Total rows</div>', unsafe_allow_html=True)
    with stat_cols[1]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n ok">{summary["ok"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Scored</div>', unsafe_allow_html=True)
    with stat_cols[2]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n error">{summary["errors"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Excluded (failed validation)</div>', unsafe_allow_html=True)

    if summary["errors"]:
        st.warning(
            f"{summary['errors']} row(s) failed validation - see the "
            "'Validation Status' / 'Validation Error' columns below. "
            "Those rows are excluded from the metrics below."
        )

    st.subheader("Predicted vs. Actual - new accuracy vs. original training metrics")
    new_metrics = compute_validation_metrics(result_df)

    rows_html = []
    for target in TARGET_PROPERTIES:
        train = training_metrics[target]
        new = new_metrics[target]
        rows_html.append(
            f"<tr><td class=\"mono\">{target}</td>"
            f"<td class=\"num\">{train['r2']:.3f}</td>"
            f"<td class=\"num\">{train['mae']:.2f}</td>"
            f"<td class=\"num\">{_fmt(new['r2'], 3)}</td>"
            f"<td class=\"num\">{_fmt(new['mae'], 2)}</td>"
            f"<td class=\"num\">{_fmt(new['rmse'], 2)}</td>"
            f"<td class=\"num\">{new['n']}</td></tr>"
        )
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table class="pp-info-table"><thead><tr>'
        "<th>Target</th><th>Training R²</th><th>Training MAE</th>"
        "<th>New R²</th><th>New MAE</th><th>New RMSE</th><th>Scored rows</th>"
        f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "\"Training\" columns are read from the model artifact's original "
        "validation metrics (see Model Information). \"New\" columns are "
        "computed live from this upload's scored rows only. R² shows as "
        "— when fewer than 2 rows were scored for that target, or "
        "when every scored Actual value for it is identical (R² is "
        "undefined in both cases)."
    )

    try:
        styled = result_df.style.apply(_highlight_status, axis=1)
        st.dataframe(styled, width="stretch")
    except Exception:
        st.dataframe(result_df, width="stretch")

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download results (CSV)",
        data=dataframe_to_csv_bytes(result_df),
        file_name="pp_validation_results.csv",
        mime="text/csv",
        width="stretch",
    )
    download_col2.download_button(
        "Download results (Excel)",
        data=dataframe_to_excel_bytes(result_df),
        file_name="pp_validation_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
