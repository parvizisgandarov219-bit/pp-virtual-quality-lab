import streamlit as st

from pages_ui import (
    batch_prediction,
    dashboard,
    model_improvement,
    model_information,
    model_validation,
    single_prediction,
)
from ui_components import (
    MODEL_PATH,
    PAGE_BATCH,
    PAGE_DASHBOARD,
    PAGE_ICON,
    PAGE_MODEL_IMPROVEMENT,
    PAGE_MODEL_INFO,
    PAGE_SINGLE,
    PAGE_TITLE,
    PAGE_VALIDATION,
    get_model_artifact,
    inject_theme_css,
    render_sidebar,
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

inject_theme_css()

try:
    artifact = get_model_artifact(MODEL_PATH)
    load_error = None
except Exception as exc:
    artifact = None
    load_error = str(exc)

render_sidebar(models_loaded=load_error is None)

if load_error:
    st.error(
        "Prediction models could not be loaded, so this tool is "
        f"currently unavailable. Details: {load_error}"
    )
    st.stop()

active_page = st.session_state.active_page

if active_page == PAGE_SINGLE:
    single_prediction.render(artifact)
elif active_page == PAGE_BATCH:
    batch_prediction.render(artifact)
elif active_page == PAGE_MODEL_INFO:
    model_information.render(artifact)
elif active_page == PAGE_VALIDATION:
    model_validation.render(artifact)
elif active_page == PAGE_MODEL_IMPROVEMENT:
    model_improvement.render(artifact)
else:
    dashboard.render(artifact)
