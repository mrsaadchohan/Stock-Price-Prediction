import joblib
from fastapi import FastAPI, HTTPException
import uvicorn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pydantic import BaseModel

app = FastAPI()

# Load trained LSTM model
model = tf.keras.models.load_model("model/stock_price.h5")

# Load scalers
feature_scaler = joblib.load("model/feature_scaler.pkl")
target_scaler = joblib.load("model/target_scaler.pkl")

# Define the number of timesteps the model expects
TIMESTEPS = 90


class StockInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float


# Store last TIMESTEPS inputs
input_history = []


@app.post("/predict/")
def predict_stock(data: StockInput):
    global input_history

    try:
        new_input = np.array([[data.open, data.high, data.low, data.volume]])

        # Scale new input using feature_scaler
        scaled_input = feature_scaler.transform(new_input)

        # Maintain TIMESTEPS history
        input_history.append(scaled_input)
        while len(input_history) < TIMESTEPS:
            input_history.insert(0, scaled_input)  # Fill missing data

        input_history = input_history[-TIMESTEPS:]

        # Convert to LSTM input shape (1, TIMESTEPS, features)
        input_array = np.array(input_history).reshape(1, TIMESTEPS, 4)

        # Predict scaled closing price
        scaled_prediction = model.predict(input_array)

        # Convert to actual closing price
        actual_prediction = target_scaler.inverse_transform(
            np.array(scaled_prediction).reshape(-1, 1))

        predicted_close = float(actual_prediction[0][0])

        return {"predicted_close": predicted_close}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
