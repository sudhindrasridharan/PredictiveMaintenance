
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

# Repository paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deployment" / "best_model.pkl"

FEATURES = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]

st.title("⚙️ Predictive Maintenance")
st.write(
    "Enter the engine sensor readings to predict the engine condition "
    "using the trained machine learning model."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

# Collect sensor inputs
st.subheader("Engine Sensor Readings")

with st.form("sensor_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        engine_rpm = st.number_input(
            "Engine rpm", value=0.0, format="%.4f"
        )
        lub_oil_pressure = st.number_input(
            "Lub oil pressure", value=0.0, format="%.4f"
        )
        fuel_pressure = st.number_input(
            "Fuel pressure", value=0.0, format="%.4f"
        )

    with col2:
        coolant_pressure = st.number_input(
            "Coolant pressure", value=0.0, format="%.4f"
        )
        lub_oil_temp = st.number_input(
            "Lub oil temp", value=0.0, format="%.4f"
        )
        coolant_temp = st.number_input(
            "Coolant temp", value=0.0, format="%.4f"
        )

    submitted = st.form_submit_button(
        "Predict Engine Condition",
        type="primary",
        use_container_width=True
    )

if submitted:
    # Create the input DataFrame with the exact training feature names
    input_data = pd.DataFrame(
        [[
            engine_rpm,
            lub_oil_pressure,
            fuel_pressure,
            coolant_pressure,
            lub_oil_temp,
            coolant_temp,
        ]],
        columns=FEATURES
    )

    st.subheader("Submitted Sensor Data")
    st.dataframe(input_data, use_container_width=True)

    try:
        # Generate prediction
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.metric("Predicted Engine Condition", str(prediction))

        # Display probabilities when supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            probability_data = pd.DataFrame({
                "Engine Condition": classes,
                "Predicted Probability": probabilities
            })

            st.write("Prediction probabilities")
            st.dataframe(
                probability_data,
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "The displayed class is the model's predicted target label. "
            "Confirm the meaning of each label using the dataset documentation."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption("Predictive Maintenance | Machine Learning Deployment")
