# src/inference/lstm_inference.py

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# ------------------------------------------------------------
# ✅ Load model and scaler
# ------------------------------------------------------------
def load_model_and_scaler(model_path, scaler_path):
    # ✅ Load compatible Keras v3 model
    model = tf.keras.models.load_model(model_path)

    # ✅ Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

# ------------------------------------------------------------
# ✅ Create sequences identical to training pipeline
# ------------------------------------------------------------
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ------------------------------------------------------------
# ✅ Predict using same logic as training pipeline
# ------------------------------------------------------------
def predict_stock(df, model, scaler, seq_len=60, split_ratio=0.9):

    close_data = df[["Close"]].values

    # Split BEFORE scaling
    train_len = int(len(close_data) * split_ratio)
    train = close_data[:train_len]
    test  = close_data[train_len:]

    # Scale using ONLY train scaler
    scaled_train = scaler.transform(train)
    scaled_test  = scaler.transform(test)

    # Create sequences
    X_train, y_train = create_sequences(scaled_train, seq_len)
    X_test,  y_test  = create_sequences(scaled_test, seq_len)

    # Predict
    pred_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(pred_scaled).squeeze()

    # Actual values (skip first seq_len)
    actual = test[seq_len:].squeeze()

    # Return all required info
    return actual, predicted, train_len, seq_len


# ------------------------------------------------------------
# ✅ Raw Close-Price Plot
# ------------------------------------------------------------
def plot_close_price(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")
    plt.title("📈 Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    return plt


# ------------------------------------------------------------
# ✅ Train / Test / Predictions Plot
# ------------------------------------------------------------
def plot_predictions_full(df, actual, predicted, train_len, seq_len):

    data = df.copy()

    train = data.iloc[:train_len]
    test  = data.iloc[train_len + seq_len:].copy()

    # Trim test to prediction size
    test = test.iloc[:len(predicted)]
    test["Predictions"] = predicted

    plt.figure(figsize=(14, 7))

    plt.plot(train["Date"], train["Close"], color="blue", label="Train (Actual)")
    plt.plot(test["Date"], test["Close"], color="orange", label="Test (Actual)")
    plt.plot(test["Date"], test["Predictions"], color="red", label="Predictions")

    plt.title("📉 NVIDIA Stock Price Prediction (Train/Test/Predictions)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)

    return plt


# ------------------------------------------------------------
# ✅ Metric Calculation
# ------------------------------------------------------------
def compute_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return rmse, mae, r2

def plot_full_actual_vs_predicted(df, actual, predicted, split_ratio=0.9, seq_len=60):
    import matplotlib.pyplot as plt

    # Copy dataframe
    data = df.copy()

    # Length of full dataset
    n = len(data)

    # TRAIN/TEST SPLIT INDEX
    train_len = int(n * split_ratio)

    # ✅ Create an empty prediction column for the entire dataset
    data["Predicted"] = np.nan

    # ✅ Predicted values begin at:
    # start = train_len + seq_len
    start_index = train_len + seq_len

    # ✅ Assign predicted values to correct date locations
    data.loc[start_index:start_index+len(predicted)-1, "Predicted"] = predicted

    # --- ✅ PLOT ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # FULL Actual Data
    ax.plot(data["Date"], data["Close"], label="Actual Close Price", color="blue")

    # FULL Predicted Data (only visible where prediction exists)
    ax.plot(data["Date"], data["Predicted"], label="Predicted Close Price", color="red")

    ax.set_title("Full Stock Price vs Predicted Price", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True)

    return fig
