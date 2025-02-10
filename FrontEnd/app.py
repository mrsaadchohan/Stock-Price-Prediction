import streamlit as st
import requests


# FastAPI backend URL
FASTAPI_URL='http://localhost:8000/predict/'
# Streamlit UI
st.title("📈 Stock Market Prediction")

# User input
st.sidebar.header("Enter Stock Data")
open_price = st.sidebar.number_input(
    "Open Price", min_value=0.0, step=0.1, format="%.2f")
high_price = st.sidebar.number_input(
    "High Price", min_value=0.0, step=0.1, format="%.2f")
low_price = st.sidebar.number_input(
    "Low Price", min_value=0.0, step=0.1, format="%.2f")
volume = st.sidebar.number_input(
    "Volume", min_value=0.0, step=100.0, format="%.2f")

# Prediction button
if st.sidebar.button("Predict"):
    input_data = {
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "volume": volume
    }

    response = requests.post(FASTAPI_URL, json=input_data)

    if response.status_code == 200:
        result = response.json()
        st.success(
            f"Predicted Closing Price: ${result['predicted_close']:.2f}")

    else:
        st.error("Prediction failed. Please check input values.")

# Footer
st.markdown("---")
st.markdown("📊 Built with **FastAPI** and **Streamlit**")
