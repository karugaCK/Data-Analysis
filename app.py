
import numpy as np
import streamlit as st
import tensorflow as tf
import joblib
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Traffic Volume Prediction (LSTM)",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Volume Prediction System")
st.write(
    "This system predicts **next-hour traffic volume** using a trained "
    "**LSTM model** based on the I-94 Traffic Dataset."
)

# ------------------------------
# Load Model & Scaler
# ------------------------------
PROJECT_DIR = "/content/drive/MyDrive/LSTM_Traffic_Prediction_Project"
MODEL_PATH = os.path.join(PROJECT_DIR, "lstm_traffic_model.keras")
SCALER_PATH = os.path.join(PROJECT_DIR, "minmax_scaler.save")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()

# ------------------------------
# Input Section
# ------------------------------
st.subheader("Input: Last 24 Hourly Traffic Volumes")

st.info("Enter traffic volume values for the previous 24 hours.")

inputs = []
cols = st.columns(4)

for i in range(24):
    with cols[i % 4]:
        value = st.number_input(
            f"Hour {i+1}",
            min_value=0.0,
            value=50.0,
            step=1.0
        )
        inputs.append(value)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Next Hour Traffic"):
    data = np.array(inputs).reshape(-1, 1)

    data_scaled = scaler.transform(data)
    X_input = data_scaled.reshape(1, 24, 1)

    prediction_scaled = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction_scaled)

    st.success(
        f"✅ **Predicted Traffic Volume (Next Hour): "
        f"{prediction[0][0]:.2f} vehicles**"
    )

# ------------------------------
# Footer
# ------------------------------
st.caption(
    "Model: LSTM | Window Size: 24 Hours | "
    "Dataset: I-94 Traffic Volume"
)
