
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PP Virtual Quality Lab",
    page_icon="🧪",
    layout="wide"
)

grade_models = joblib.load(
    "virtual_lab_models.joblib"
)
st.write(grade_models.keys())
GRADE_OPTIONS = [
    "CA0900BM",
    "CB0900MO",
    "CB1248MO",
    "CB1640MO",
    "CB1849MO",
    "CB2000GT",
    "CB3000GT",
    "CB3648MO",
    "CB4048MO",
    "CB4848MO",
    "CB6448MO",
    "CB8248MO"
]

st.title("PP Virtual Quality Lab")

st.info(
    "Decision-support model only. "
    "Laboratory confirmation remains required."
)

left, middle1, middle2, right = st.columns(4)

with left:
    grade = st.selectbox(
        "Grade",
        GRADE_OPTIONS,
        index=9
    )

with middle1:
    mfr = st.number_input(
        "MFR, g/10 min",
        min_value=0.0,
        value=48.0,
        step=0.1
    )

with middle2:
    xs = st.number_input(
        "XS, wt%",
        min_value=0.0,
        value=16.0,
        step=0.1
    )

with right:
    c2 = st.number_input(
        "C2, wt%",
        min_value=0.0,
        value=8.6,
        step=0.1
    )

if st.button(
    "Calculate Properties",
    type="primary",
    use_container_width=True
):

    new_product = pd.DataFrame({
        "MFR": [mfr],
        "XS": [xs],
        "C2": [c2],
        "Grade": [grade]
    })

    impact = grade_models[
        "Izod Impact"
    ].predict(new_product)[0]

    tensile = grade_models[
        "Tensile Modulus"
    ].predict(new_product)[0]

    flexural = grade_models[
        "Flexural Modulus"
    ].predict(new_product)[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Izod Impact",
        f"{impact:.2f}"
    )

    col2.metric(
        "Predicted Tensile Modulus",
        f"{tensile:.0f} MPa"
    )

    col3.metric(
        "Predicted Flexural Modulus",
        f"{flexural:.0f} MPa"
    )

st.divider()

st.subheader("Model Validation")

v1, v2, v3 = st.columns(3)

v1.metric("Izod Impact R²", "0.977")
v2.metric("Tensile Modulus R²", "0.587")
v3.metric("Flexural Modulus R²", "0.611")

st.warning(
    "Impact is the strongest validated prediction. "
    "Tensile and Flexural are early estimates and "
    "must not be used alone for product release."
)

