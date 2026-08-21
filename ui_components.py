"""Shared Streamlit presentation helpers for the PP Virtual Quality Lab UI.

Unlike model_utils.py / batch_utils.py (deliberately Streamlit-free so
they're fast and simple to unit test), this module imports streamlit and
holds only rendering, UI-layer caching, and small display helpers. No
prediction or validation logic lives here - anything that decides a
predicted value or whether a row is valid belongs in model_utils.py or
batch_utils.py instead.
"""

from pathlib import Path

import streamlit as st

from model_utils import SUPPORTED_GRADES, load_model_artifact

APP_ROOT = Path(__file__).parent
MODEL_PATH = APP_ROOT / "pp_virtual_lab_models .joblib"
# Best-effort local audit log of predictions made. NOTE: on Streamlit
# Community Cloud the filesystem is ephemeral, so this file will not
# survive an app restart/redeploy there - see README for details.
PREDICTION_LOG_PATH = APP_ROOT / "predictions_log.csv"

PAGE_ICON = "🧪"
PAGE_TITLE = "PP Virtual Lab"

# Presentational only: which R² earns a target the "Strong fit" badge
# versus "Moderate fit". R² is a goodness-of-fit statistic, not a
# probability or a confidence interval, so the UI deliberately avoids
# calling it "confidence" anywhere - see performance_tier() below.
STRONG_FIT_R2_THRESHOLD = 0.9

PAGE_DASHBOARD = "dashboard"
PAGE_SINGLE = "single"
PAGE_BATCH = "batch"
PAGE_MODEL_INFO = "model_info"
NAV_ITEMS = [
    (PAGE_DASHBOARD, "Dashboard"),
    (PAGE_SINGLE, "Single Prediction"),
    (PAGE_BATCH, "Batch Prediction"),
    (PAGE_MODEL_INFO, "Model Information"),
]


@st.cache_resource(show_spinner="Loading prediction models...")
def get_model_artifact(path: Path) -> dict:
    return load_model_artifact(path)


def performance_tier(r2: float) -> tuple[str, str]:
    """Return (label, css_key) classifying a target's R² for display.

    Purely presentational classification of an existing, already-computed
    R² value - it does not change how R² is calculated or displayed
    elsewhere (the exact R² number is always shown alongside the badge).
    Labeled in terms of model fit, not "confidence" - R² measures how
    well the model's predictions match validation data, not a
    probability or a confidence interval.
    """
    if r2 >= STRONG_FIT_R2_THRESHOLD:
        return "Strong fit", "high"
    return "Moderate fit", "moderate"


CSS = """
<style>
:root {
    --pp-ground: #F4F6F7;
    --pp-surface: #FFFFFF;
    --pp-ink: #16212B;
    --pp-ink-muted: #5B6B76;
    --pp-ink-faint: #8C99A2;
    --pp-line: #D8DEE2;
    --pp-accent: #0E5C73;
    --pp-accent-soft: #E4EEF0;
    --pp-accent-run: #D9622B;
    --pp-ok: #2E7D4F;
    --pp-ok-bg: #E4F1E9;
    --pp-warn: #96650A;
    --pp-warn-bg: #FBF0DD;
    --pp-danger: #B23B3B;
    --pp-danger-bg: #F8E6E6;
}
@media (prefers-color-scheme: dark) {
    :root {
        --pp-ground: #10171E;
        --pp-surface: #17212A;
        --pp-ink: #E7EDF0;
        --pp-ink-muted: #93A3AC;
        --pp-ink-faint: #66767F;
        --pp-line: #29343E;
        --pp-accent: #52B7CB;
        --pp-accent-soft: #17323A;
        --pp-accent-run: #EE8148;
        --pp-ok: #55B583;
        --pp-ok-bg: #14261C;
        --pp-warn: #D9A44A;
        --pp-warn-bg: #2A2110;
        --pp-danger: #E08585;
        --pp-danger-bg: #2C1717;
    }
}

[data-testid="stSidebar"] { border-right: 1px solid var(--pp-line); }

.stButton > button[kind="primary"] {
    background-color: var(--pp-accent-run);
    border-color: var(--pp-accent-run);
    font-weight: 700;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.06);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(22, 33, 43, 0.06);
}

.pp-brand { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 6px; }
.pp-brand .mark {
    flex: none; width: 32px; height: 32px; border-radius: 7px;
    background: var(--pp-accent); color: #fff;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.pp-brand .word { font-weight: 700; font-size: 15px; letter-spacing: 0.02em; line-height: 1.2; color: var(--pp-ink); }
.pp-brand .tag { font-size: 11px; color: var(--pp-ink-muted); margin-top: 2px; }

.pp-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;
    padding: 4px 9px; border-radius: 999px;
    background: var(--pp-accent-soft); color: var(--pp-accent);
    margin-bottom: 10px;
}

.pp-tier-pill { font-size: 10px; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
    padding: 3px 8px; border-radius: 999px; white-space: nowrap; }
.pp-tier-pill.high { background: var(--pp-ok-bg); color: var(--pp-ok); }
.pp-tier-pill.moderate { background: var(--pp-warn-bg); color: var(--pp-warn); }

.pp-status-pill { font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 999px; white-space: nowrap; }
.pp-status-pill.ok { background: var(--pp-ok-bg); color: var(--pp-ok); }
.pp-status-pill.error { background: var(--pp-danger-bg); color: var(--pp-danger); }

.pp-card-label { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--pp-ink-muted); margin-bottom: 4px; }
.pp-card-value { font-variant-numeric: tabular-nums; font-size: 30px; font-weight: 700; line-height: 1.1; color: var(--pp-ink); }
.pp-card-value .unit { font-size: 13px; font-weight: 500; color: var(--pp-ink-muted); margin-left: 6px; }
.pp-card-foot { margin-top: 8px; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.pp-card-r2 { font-size: 11px; color: var(--pp-ink-faint); font-variant-numeric: tabular-nums; }

.pp-stat-n { font-variant-numeric: tabular-nums; font-size: 24px; font-weight: 700; line-height: 1; }
.pp-stat-l { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--pp-ink-muted); margin-top: 4px; }
.pp-stat-n.ok { color: var(--pp-ok); }
.pp-stat-n.error { color: var(--pp-danger); }

table.pp-info-table { border-collapse: collapse; width: 100%; font-size: 13px; min-width: 460px; }
table.pp-info-table th {
    text-align: left; font-size: 10.5px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--pp-ink-muted); padding: 8px 12px; border-bottom: 1px solid var(--pp-line); white-space: nowrap;
}
table.pp-info-table td { padding: 8px 12px; border-bottom: 1px solid var(--pp-line); color: var(--pp-ink); }
table.pp-info-table tr:last-child td { border-bottom: none; }
table.pp-info-table td.num { font-variant-numeric: tabular-nums; }
table.pp-info-table td.mono { font-family: ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono", Consolas, monospace; }

.pp-flow { display: flex; align-items: stretch; gap: 6px; flex-wrap: wrap; margin: 4px 0 2px; }
.pp-flow-step {
    flex: 1 1 150px; min-width: 140px; border: 1px solid var(--pp-line); border-radius: 6px;
    padding: 10px 12px; background: var(--pp-surface);
}
.pp-flow-step .t { font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--pp-ink-muted); }
.pp-flow-step .d { font-size: 12.5px; font-weight: 600; color: var(--pp-ink); margin-top: 4px; line-height: 1.3; }
.pp-flow-arrow { display: flex; align-items: center; justify-content: center; color: var(--pp-accent); font-size: 16px; flex: 0 0 20px; }

.pp-io-table td.yes { color: var(--pp-ok); font-weight: 700; }
.pp-io-table td.no { color: var(--pp-danger); font-weight: 700; }
</style>
"""


def inject_theme_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_sidebar(models_loaded: bool) -> None:
    """Render the persistent sidebar: brand, navigation, and status footer.

    Navigation is implemented as sidebar buttons that set
    st.session_state["active_page"], rather than st.navigation/st.Page,
    so that the whole app stays a single-script rerun and every
    navigation path is exercised the same way pytest's AppTest already
    exercises the rest of the app (button.click().run()).
    """
    if "active_page" not in st.session_state:
        st.session_state.active_page = PAGE_DASHBOARD

    with st.sidebar:
        st.markdown(
            '<div class="pp-brand">'
            '<div class="mark">\U0001F9EA</div>'
            '<div><div class="word">PP VIRTUAL LAB</div>'
            '<div class="tag">Polypropylene property prediction</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<span class="pp-badge">Decision-support tool</span>', unsafe_allow_html=True)

        for page_key, label in NAV_ITEMS:
            is_active = st.session_state.active_page == page_key
            if st.button(
                label,
                key=f"nav_{page_key}",
                type="primary" if is_active else "secondary",
                width="stretch",
            ):
                st.session_state.active_page = page_key

        st.divider()
        if models_loaded:
            st.caption("✅ Models loaded")
        else:
            st.caption("❌ Model load failed")
        st.caption(f"{len(SUPPORTED_GRADES)} of 31 trained grades exposed")
