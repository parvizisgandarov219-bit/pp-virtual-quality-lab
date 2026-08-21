"""Dashboard / Home page."""

import streamlit as st

from model_utils import SUPPORTED_GRADES, TARGET_PROPERTIES
from ui_components import PAGE_BATCH, PAGE_SINGLE, performance_tier

TOTAL_TRAINED_GRADES = 31


def render(artifact: dict) -> None:
    metrics = artifact["metrics"]

    st.markdown('<span class="pp-badge">Overview</span>', unsafe_allow_html=True)
    st.title("Dashboard")
    st.caption("Model status and performance at a glance, before you run a prediction.")

    st.info(
        "Decision-support predictions only. Laboratory confirmation is "
        "required before product release."
    )

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        with st.container(border=True):
            st.markdown('<div class="pp-card-label">Model status</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="pp-card-value" style="font-size:20px;">✅ Loaded</div>',
                unsafe_allow_html=True,
            )
    with status_col2:
        with st.container(border=True):
            st.markdown('<div class="pp-card-label">Grades exposed</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="pp-card-value" style="font-size:20px;">'
                f'{len(SUPPORTED_GRADES)}<span class="unit">of {TOTAL_TRAINED_GRADES} trained</span></div>',
                unsafe_allow_html=True,
            )

    st.subheader("Model Performance")
    cols = st.columns(len(TARGET_PROPERTIES))
    for col, target in zip(cols, TARGET_PROPERTIES):
        target_metrics = metrics[target]
        r2 = target_metrics["r2"]
        mae = target_metrics["mae"]
        rows = target_metrics["rows"]
        tier_label, tier_key = performance_tier(r2)

        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:flex-start;gap:8px;">'
                    f'<div class="pp-card-label" style="margin-bottom:0;">{target}</div>'
                    f'<span class="pp-tier-pill {tier_key}">{tier_label}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="pp-card-value" style="margin-top:8px;">{r2:.3f}'
                    f'<span class="unit">R²</span></div>',
                    unsafe_allow_html=True,
                )
                st.progress(min(max(r2, 0.0), 1.0))
                st.caption(f"MAE {mae:.2f} · {rows} validation rows")

    st.subheader("Quick actions")
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        with st.container(border=True):
            st.markdown("**Run a single prediction**")
            st.caption("One grade, one formulation, three properties.")
            if st.button("Go to Single Prediction", key="dash_go_single", width="stretch"):
                st.session_state.active_page = PAGE_SINGLE
                st.rerun()
    with action_col2:
        with st.container(border=True):
            st.markdown("**Upload a batch file**")
            st.caption("CSV or Excel, any number of rows.")
            if st.button("Go to Batch Prediction", key="dash_go_batch", width="stretch"):
                st.session_state.active_page = PAGE_BATCH
                st.rerun()

    with st.container(border=True):
        head_col, meta_col = st.columns([3, 1])
        head_col.markdown("**Supported grades**")
        meta_col.caption(f"{len(SUPPORTED_GRADES)} grades")
        st.markdown(
            " ".join(
                f'<code style="margin-right:4px;">{grade}</code>'
                for grade in SUPPORTED_GRADES
            ),
            unsafe_allow_html=True,
        )
