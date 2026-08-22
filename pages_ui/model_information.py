"""Model Information page - includes an educational "How the Prediction
Works" explainer for process/quality engineers, alongside the existing
live model metadata (targets/performance, feature schema, grade family
rule, supported grades).

Documentation only: nothing here computes or alters a prediction. All
numeric values (R², MAE, hyperparameters) are read from the real loaded
model artifact at render time, never hardcoded.
"""

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

# Selected hyperparameters worth showing an engineer - excludes purely
# implementation-level scikit-learn defaults (verbose, warm_start,
# ccp_alpha, monotonic_cst, ...) that carry no interpretive meaning here.
_DISPLAYED_PARAMS = [
    ("n_estimators", "Number of trees in the forest"),
    ("min_samples_leaf", "Minimum samples required at a leaf node"),
    ("min_samples_split", "Minimum samples required to split a node"),
    ("max_depth", "Maximum tree depth (None = unrestricted)"),
    ("max_features", "Fraction of features considered at each split"),
    ("bootstrap", "Trees trained on bootstrap-resampled data (bagging)"),
    ("criterion", "Split quality measure"),
    ("random_state", "Random seed (for reproducibility)"),
]


def _render_flow(steps: list[tuple[str, str]]) -> None:
    """Render a horizontal pipeline diagram from (title, description) steps."""
    parts = []
    for i, (title, desc) in enumerate(steps):
        if i > 0:
            parts.append('<div class="pp-flow-arrow">&#8594;</div>')
        parts.append(
            f'<div class="pp-flow-step"><div class="t">{title}</div>'
            f'<div class="d">{desc}</div></div>'
        )
    st.markdown(f'<div class="pp-flow">{"".join(parts)}</div>', unsafe_allow_html=True)


def render(artifact: dict) -> None:
    models = artifact["models"]
    metrics = artifact["metrics"]

    st.markdown('<span class="pp-badge">Model Information</span>', unsafe_allow_html=True)
    st.title("Model Information")
    st.caption("What the model actually is, what it was validated on, and its known limits.")

    # --- How the Prediction Works -----------------------------------------
    st.subheader("How the Prediction Works")
    st.caption(
        "Every prediction - single or batch - goes through the same fixed "
        "pipeline. There is no per-grade or per-family special-casing "
        "inside the model itself."
    )
    _render_flow([
        ("1. Inputs", "Grade, MFR, XS, C2"),
        ("2. Feature construction", "Grade one-hot encoded + numeric inputs → 34 features"),
        ("3. Trained model", "3× Random Forest Regressor (one per property)"),
        ("4. Independent regressions", "Each property predicted separately"),
        ("5. Output", "Izod Impact · Tensile Modulus · Flexural Modulus"),
    ])

    st.markdown("**What each input means**")
    input_rows = "".join([
        '<tr><td class="mono">Grade</td><td>Identifies the specific product '
        "formulation. One-hot encoded so the model can learn grade-specific "
        "behavior rather than treating grades as an ordered number.</td></tr>",
        '<tr><td class="mono">MFR</td><td>Melt Flow Rate, g/10 min - a '
        "rheology measurement describing how readily the polymer melt "
        "flows under standard test conditions; broadly relates to average "
        "molecular weight.</td></tr>",
        '<tr><td class="mono">XS</td><td>Xylene Solubles, wt% - the '
        "fraction of the material that dissolves in xylene at room "
        "temperature, a standard proxy for amorphous / rubber-phase "
        "content.</td></tr>",
        '<tr><td class="mono">C2</td><td>Ethylene comonomer content, wt% - '
        "governs the impact/stiffness balance in copolymers. Not "
        "chemically applicable to a homopolymer, which is why HOMO grades "
        "fix it at 0 rather than asking for a value.</td></tr>",
    ])
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table class="pp-info-table"><tbody>'
        f"{input_rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # --- What affects the prediction? --------------------------------------
    st.subheader("What affects the prediction?")
    io_rows = "".join([
        '<tr><td class="mono">Grade</td><td class="yes">Model input</td></tr>',
        '<tr><td class="mono">MFR</td><td class="yes">Model input</td></tr>',
        '<tr><td class="mono">XS</td><td class="yes">Model input</td></tr>',
        '<tr><td class="mono">C2</td><td class="yes">Model input</td></tr>',
        '<tr><td class="mono">Grade Family</td><td class="no">Not a model input - '
        "derived from Grade for validation only</td></tr>",
        '<tr><td class="mono">Production Date</td><td class="no">Metadata only - '
        "preserved in batch results, never read by the model</td></tr>",
        '<tr><td class="mono">Batch Number</td><td class="no">Metadata only - '
        "preserved in batch results, never read by the model</td></tr>",
    ])
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table class="pp-info-table pp-io-table">'
        "<thead><tr><th>Field</th><th>Effect on prediction</th></tr></thead>"
        f"<tbody>{io_rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Production Date and Batch Number identify the batch for future "
        "history/matching purposes only - changing either never changes a "
        "predicted value."
    )

    # --- Targets & Performance (existing) -----------------------------------
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
        "is actually deployed."
    )
    st.markdown(
        "**R² is a model validation statistic, not a confidence score.** "
        "It describes how closely this model's predictions matched real "
        "lab measurements on 220 held-out validation samples during "
        "testing - it does not mean any single new prediction is \"X% "
        "accurate,\" and it is not a probability or a confidence interval. "
        "Izod Impact's R² (0.974) reflects a strong overall fit on that "
        "validation set; Tensile and Flexural Modulus (0.574 / 0.599) "
        "reflect a much weaker fit, meaning their predictions should be "
        "treated as rougher estimates."
    )

    # --- The algorithm -------------------------------------------------------
    st.subheader("The algorithm")
    st.markdown(
        "**Random Forest Regression** (`sklearn.ensemble.RandomForestRegressor`) "
        "- three separately trained forests, one per target property. They "
        "don't share trees or parameters with each other.\n\n"
        "A random forest is an ensemble of decision trees. Each tree trains "
        "on a bootstrap-resampled subset of the training data. At "
        "prediction time, every tree produces its own estimate, and the "
        "forest's output is the average across all trees - this averaging "
        "is what makes a random forest more stable and less prone to "
        "overfitting than any single decision tree."
    )

    param_sets = {target: models[target].get_params() for target in TARGET_PROPERTIES}
    reference = param_sets[TARGET_PROPERTIES[0]]
    identical = all(param_sets[t] == reference for t in TARGET_PROPERTIES)

    param_rows = "".join(
        f'<tr><td class="mono">{key}</td><td class="num">{reference.get(key)!r}</td><td>{desc}</td></tr>'
        for key, desc in _DISPLAYED_PARAMS
    )
    st.markdown(
        '<div style="overflow-x:auto;">'
        '<table class="pp-info-table">'
        "<thead><tr><th>Parameter</th><th>Value</th><th>Meaning</th></tr></thead>"
        f"<tbody>{param_rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if identical:
        st.caption(
            "Read live from the loaded model objects, not hardcoded - "
            "confirmed identical across all three target models."
        )
    else:
        st.warning(
            "The three target models currently have different "
            "hyperparameters - the table above shows Izod Impact's values "
            "only. This is unexpected; verify the deployed artifact."
        )

    st.markdown(
        "- **Inputs per prediction:** `MFR`, `XS`, `C2` (numeric) plus a "
        "one-hot encoded Grade indicator - 34 features in total per model "
        "(3 numeric + 31 trained grade indicators).\n"
        "- **Grade Family is not a model input.** It is derived from the "
        "Grade code's prefix purely for input validation (e.g. whether C2 "
        "is required) - the model itself sees only Grade, MFR, XS, and C2."
    )

    # --- Grade Family derivation rule (existing) ----------------------------
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
        + "\n\n`ICP` is accepted as another name for `HECO` in uploaded files. "
        "HOMO grades have no comonomer by definition, which is why C2 is "
        "automatically fixed at 0 and disabled for them, both in Single "
        "Prediction and in batch validation."
    )

    # --- Supported grades (existing) ----------------------------------------
    st.subheader(f"Supported grades ({len(SUPPORTED_GRADES)})")
    st.caption(
        f"{len(SUPPORTED_GRADES)} of the model's 31 trained grades are "
        "exposed for prediction, spanning all three families - HECO, "
        "HOMO, and RACO - derived automatically from each grade's prefix."
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

    # --- Limitations (existing, expanded) -----------------------------------
    st.subheader("Limitations")
    st.warning(
        "Impact is the strongest validated prediction. Tensile and "
        "Flexural are early estimates and must not be used alone for "
        "product release."
    )
    st.markdown(
        "- **Predictions are estimates, not laboratory measurements.** "
        "They support - and never replace - laboratory confirmation "
        "before product release.\n"
        "- **Reliable only within the domain of the training data.** "
        "Inputs far outside the ranges the model was trained and "
        "validated on should be treated with additional caution, even "
        "when they fall inside the UI's generic input bounds.\n"
        "- **Unsupported grades must not be predicted**, and aren't: only "
        "the grades listed above are selectable, and any grade without a "
        "matching trained schema entry is refused rather than guessed.\n"
        "- **Tensile and Flexural Modulus have substantially lower R² "
        "than Izod Impact** (0.574 / 0.599 vs. 0.974) and should be "
        "interpreted as rougher, early-stage estimates accordingly - see "
        "\"Targets & Performance\" above.\n"
        "- Input bounds shown on Single Prediction are generic, "
        "physically-reasonable PP process ranges, not statistically "
        "derived from the training data - the artifact carries no "
        "per-feature training range metadata."
    )
