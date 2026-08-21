"""Single Prediction page."""

import streamlit as st

from model_utils import (
    C2_BOUNDS,
    FAMILY_HOMO,
    FAMILY_LABELS,
    IZOD_IMPACT_UNIT,
    MFR_BOUNDS,
    SUPPORTED_GRADES,
    XS_BOUNDS,
    derive_grade_family,
    log_prediction,
    predict_all,
    resolve_c2_for_family,
)
from ui_components import PREDICTION_LOG_PATH, performance_tier


def _family_display(grade: str) -> tuple[str, str, bool]:
    """Given a grade, return (family, label, c2_disabled) for the UI.

    Pulled out as its own function so the derivation and HOMO-C2-disable
    logic is unit-testable even for grades that aren't currently
    selectable in the live Grade dropdown - every SUPPORTED_GRADES entry
    today starts with "C" (derives to HECO), so the HOMO/RACO branches
    can only be exercised this way until a HOMO/RACO grade is exposed
    there (see tests/test_ui_components.py).
    """
    family = derive_grade_family(grade)
    return family, FAMILY_LABELS[family], family == FAMILY_HOMO


def render(artifact: dict) -> None:
    models = artifact["models"]
    feature_columns = artifact["feature_columns"]
    metrics = artifact["metrics"]

    st.markdown('<span class="pp-badge">Single Prediction</span>', unsafe_allow_html=True)
    st.title("Predict one formulation")
    st.caption(
        "Predicts Izod Impact, Tensile Modulus, and Flexural Modulus for a "
        "single grade and process input."
    )

    with st.container(border=True):
        c1, c2_col, c3, c4 = st.columns(4)

        with c1:
            grade = st.selectbox("Grade", SUPPORTED_GRADES, index=8)

        family, family_label, is_homo = _family_display(grade)

        with c2_col:
            st.text_input(
                "Grade family",
                value=family_label,
                disabled=True,
                help="Automatically determined from the grade code and cannot be edited here.",
            )

        with c3:
            mfr = st.number_input(
                "MFR, g/10 min",
                min_value=MFR_BOUNDS[0],
                max_value=MFR_BOUNDS[1],
                value=48.0,
                step=0.1,
                help=f"Typical PP melt flow rate range: {MFR_BOUNDS[0]:g}-{MFR_BOUNDS[1]:g} g/10 min.",
            )

        with c4:
            xs = st.number_input(
                "XS, wt%",
                min_value=XS_BOUNDS[0],
                max_value=XS_BOUNDS[1],
                value=16.0,
                step=0.1,
                help=f"Xylene solubles, typical range: {XS_BOUNDS[0]:g}-{XS_BOUNDS[1]:g} wt%.",
            )

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            c2 = st.number_input(
                "C2, wt%",
                min_value=C2_BOUNDS[0],
                max_value=C2_BOUNDS[1],
                value=0.0 if is_homo else 8.6,
                step=0.1,
                disabled=is_homo,
                help=(
                    "Not required for HOMO - defaulted to 0."
                    if is_homo
                    else f"Ethylene comonomer content, typical range: {C2_BOUNDS[0]:g}-{C2_BOUNDS[1]:g} wt%."
                ),
            )
        with c8:
            run_clicked = st.button("Run Prediction", type="primary", width="stretch")

    if run_clicked:
        resolved_c2 = resolve_c2_for_family(family, c2)
        try:
            predictions = predict_all(models, feature_columns, grade, mfr, xs, resolved_c2)
        except Exception as exc:
            st.error(f"Could not compute predictions: {exc}")
        else:
            result_specs = [
                ("Izod Impact", predictions["Izod Impact"], IZOD_IMPACT_UNIT, ".2f"),
                ("Tensile Modulus", predictions["Tensile Modulus"], "MPa", ".0f"),
                ("Flexural Modulus", predictions["Flexural Modulus"], "MPa", ".0f"),
            ]
            cols = st.columns(3)
            for col, (label, value, unit, fmt) in zip(cols, result_specs):
                r2 = metrics[label]["r2"]
                tier_label, tier_key = performance_tier(r2)
                with col:
                    with st.container(border=True):
                        st.markdown(f'<div class="pp-card-label">{label}</div>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="pp-card-value">{value:{fmt}}'
                            f'<span class="unit">{unit}</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="pp-card-foot">'
                            f'<span class="pp-tier-pill {tier_key}">{tier_label}</span>'
                            f'<span class="pp-card-r2">R² {r2:.3f}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.warning(
                "Impact is the strongest validated prediction. "
                "Tensile and Flexural are early estimates and "
                "must not be used alone for product release."
            )

            try:
                log_prediction(PREDICTION_LOG_PATH, grade, mfr, xs, resolved_c2, predictions)
            except Exception:
                # Logging is best-effort only and must never break the UI.
                pass
