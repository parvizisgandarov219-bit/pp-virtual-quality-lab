"""Model Improvement / Retraining roadmap page.

This page is a controlled, informational roadmap for a FUTURE model
retraining capability - it does NOT retrain, refit, or otherwise modify
the deployed model in any way, and there is no automatic retraining path
anywhere in this app. Today it only surfaces how much validated lab data
has been accumulated so far (via Model Validation's "Save to training
archive" action) and documents the intended future workflow.

Intended future workflow (not implemented here):
  1. Historical training data (not included in this repository) plus the
     accumulated lab data shown on this page are combined.
  2. A candidate model is trained from that combined dataset, separately
     from - and without ever touching - the current production
     `.joblib` artifact.
  3. The candidate's validation metrics (R² / MAE / RMSE) are compared
     side by side against the current production model's metrics.
  4. Only after explicit human review and approval would the candidate
     ever replace the production model - never automatically.
"""

import streamlit as st

from model_utils import TARGET_PROPERTIES
from ui_components import LAB_ARCHIVE_PATH
from validation_utils import read_lab_archive, summarize_lab_archive


def render(artifact: dict) -> None:
    metrics = artifact["metrics"]

    st.markdown('<span class="pp-badge">Model Improvement</span>', unsafe_allow_html=True)
    st.title("Model Improvement (Roadmap)")
    st.caption(
        "A controlled, future capability for retraining the model from "
        "accumulated lab data. Nothing on this page trains, retrains, "
        "or modifies the deployed model."
    )
    st.info(
        "Not yet active. This page only shows accumulated lab data and "
        "documents the intended workflow below - no retraining ever "
        "runs automatically, and the production model is never replaced "
        "without explicit human review and approval."
    )

    archive_df = read_lab_archive(LAB_ARCHIVE_PATH)
    summary = summarize_lab_archive(archive_df)

    st.subheader("Accumulated lab data")
    stat_cols = st.columns(3)
    with stat_cols[0]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n">{summary["rows"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Validated rows saved</div>', unsafe_allow_html=True)
    with stat_cols[1]:
        with st.container(border=True):
            st.markdown(f'<div class="pp-stat-n">{summary["unique_grades"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Distinct grades represented</div>', unsafe_allow_html=True)
    with stat_cols[2]:
        with st.container(border=True):
            date_range = summary["date_range"]
            value = f"{date_range[0]} → {date_range[1]}" if date_range else "—"
            st.markdown(f'<div class="pp-stat-n" style="font-size:16px;">{value}</div>', unsafe_allow_html=True)
            st.markdown('<div class="pp-stat-l">Production date range</div>', unsafe_allow_html=True)

    if summary["rows"] == 0:
        st.caption(
            "No lab data has been saved to the archive yet. Go to Model "
            "Validation, upload a lab results file, run validation, and "
            "use \"Save to training archive\" to start accumulating "
            "data here."
        )
    else:
        with st.expander(f"Preview accumulated data ({summary['rows']} rows)", expanded=False):
            st.dataframe(archive_df, width="stretch")

    st.subheader("Current production model")
    rows_html = "".join(
        f'<tr><td class="mono">{target}</td>'
        f'<td class="num">{metrics[target]["r2"]:.3f}</td>'
        f'<td class="num">{metrics[target]["mae"]:.2f}</td>'
        f'<td class="num">{metrics[target]["rows"]}</td></tr>'
        for target in TARGET_PROPERTIES
    )
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table class="pp-info-table"><thead><tr>'
        "<th>Target</th><th>R²</th><th>MAE</th><th>Training rows</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "These are the current production model's own training-time "
        "metrics (see Model Information) - shown here for future "
        "comparison against a retrained candidate."
    )

    st.subheader("Planned retraining workflow")
    st.markdown(
        "1. **Combine data** - the original historical training data "
        "(not included in this repository) plus the accumulated lab "
        "data shown above.\n"
        "2. **Train a candidate model** - separately, without touching "
        "the current production `.joblib` artifact in any way.\n"
        "3. **Compare metrics** - the candidate's R² / MAE / RMSE "
        "against the production model's, side by side.\n"
        "4. **Human review and approval** - only after explicit sign-off "
        "would the candidate ever replace the production model. There "
        "is no automatic replacement path, now or planned."
    )
    st.warning(
        "This capability is not implemented yet. No control on this "
        "page trains anything - accumulating validated lab data (via "
        "Model Validation) is the only active step today."
    )
