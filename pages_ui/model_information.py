"""Model Information page."""

import streamlit as st

from model_utils import (
    FAMILY_LABELS,
    GRADE_FAMILY_PREFIX_MAP,
    IZOD_IMPACT_UNIT,
    SUPPORTED_GRADES,
    TARGET_PROPERTIES,
    derive_grade_family,
)
from ui_components import performance_tier

_UNITS = {
    "Izod Impact": IZOD_IMPACT_UNIT,
    "Tensile Modulus": "MPa",
    "Flexural Modulus": "MPa",
}


def render(artifact: dict) -> None:
    metrics = artifact["metrics"]

    st.markdown('<span class="pp-badge">Model Information</span>', unsafe_allow_html=True)
    st.title("Model Information")
    st.caption("What the model actually is, what it was validated on, and its known limits.")

    st.subheader("Targets & Performance")
    rows_html = []
    for target in TARGET_PROPERTIES:
        m = metrics[target]
        tier_label, tier_key = performance_tier(m["r2"])
        rows_html.append(
            f'<tr><td>{target}</td><td>{_UNITS[target]}</td>'
            f'<td class="num">{m["r2"]:.3f}</td><td class="num">{m["mae"]:.2f}</td>'
            f'<td class="num">{m["rows"]}</td>'
            f'<td><span class="pp-tier-pill {tier_key}">{tier_label}</span></td></tr>'
        )
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table class="pp-info-table">'
        "<thead><tr><th>Target</th><th>Unit</th><th class=\"num\">R²</th>"
        '<th class="num">MAE</th><th class="num">Validation rows</th><th>Fit</th></tr></thead>'
        f"<tbody>{''.join(rows_html)}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "R² and MAE are read directly from the trained model artifact at "
        "load time, not hardcoded - they always reflect whatever artifact "
        "is actually deployed. R² measures goodness of fit on validation "
        "data; it is not a probability or a confidence interval."
    )

    st.subheader("Model type & feature schema")
    st.markdown(
        "- **Algorithm:** `RandomForestRegressor` (scikit-learn), one independently "
        "trained model per target property.\n"
        "- **Inputs per prediction:** `MFR`, `XS`, `C2` (numeric) plus a one-hot "
        "encoded Grade indicator - 34 features in total per model "
        "(3 numeric + 31 trained grade indicators).\n"
        "- **Grade Family** is not a model input. It is derived in the UI "
        "layer from the Grade code's prefix, used only for input validation "
        "(e.g. whether C2 is required) - the model itself sees only Grade, "
        "MFR, XS, and C2."
    )

    st.subheader("Grade Family derivation rule")
    st.markdown(
        "Grade Family is derived from the grade code's first letter, never "
        "freely chosen:\n\n"
        "| Prefix | Family |\n"
        "|---|---|\n"
        + "\n".join(
            f"| {prefix}* | {FAMILY_LABELS[family]} |"
            for prefix, family in GRADE_FAMILY_PREFIX_MAP.items()
        )
        + "\n\n`ICP` is accepted as another name for `HECO` in uploaded files."
    )

    st.subheader(f"Supported grades ({len(SUPPORTED_GRADES)})")
    st.caption(
        f"{len(SUPPORTED_GRADES)} of the model's 31 trained grades are "
        "exposed for prediction. Every exposed grade today starts with "
        "\"C\", so every one derives to HECO."
    )
    grade_rows = "".join(
        f'<tr><td class="mono">{grade}</td><td>{FAMILY_LABELS[derive_grade_family(grade)]}</td></tr>'
        for grade in SUPPORTED_GRADES
    )
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table class="pp-info-table">'
        "<thead><tr><th>Grade</th><th>Derived Grade Family</th></tr></thead>"
        f"<tbody>{grade_rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Limitations")
    st.warning(
        "Impact is the strongest validated prediction. Tensile and "
        "Flexural are early estimates and must not be used alone for "
        "product release."
    )
    st.markdown(
        "- Decision-support only - laboratory confirmation is required "
        "before product release.\n"
        "- Input bounds shown on Single Prediction are generic, "
        "physically-reasonable PP process ranges, not statistically "
        "derived from the training data - the artifact carries no "
        "per-feature training range metadata.\n"
        "- Only the grades listed above are selectable; predictions are "
        "refused for any grade without a matching trained schema entry."
    )
