import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_data
from src.data.preprocess.preprocess_stock import preprocess_stock

from serving.inference import (
    load_model_and_scaler,
    predict_stock,
    plot_close_price,
    plot_predictions_full,
    compute_metrics,
    plot_full_actual_vs_predicted
)

# --------------------------------------------------------
# ✅ Dashboard Title
# --------------------------------------------------------
st.title("📊 NVIDIA Stock Prediction Dashboard (LSTM Model)")

# --------------------------------------------------------
# ✅ Paths
# --------------------------------------------------------
MODEL_PATH  = os.path.join("models", "lstm_stock_model.keras")
SCALER_PATH = os.path.join("models", "standard_scaler.pkl")
DATA_PATH   = os.path.join("news_data", "nvidia_stock.csv")

# --------------------------------------------------------
# ✅ Load & preprocess data
# --------------------------------------------------------
df = load_data(DATA_PATH)
df = preprocess_stock(df)

df = df.sort_values("Date").reset_index(drop=True)

# --------------------------------------------------------
# ✅ Load model + scaler
# --------------------------------------------------------
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

# --------------------------------------------------------
# ✅ Predictions
# --------------------------------------------------------
actual, predicted, train_len, seq_len = predict_stock(df, model, scaler)

# --------------------------------------------------------
# ✅ Metrics
# --------------------------------------------------------
rmse, mae, r2 = compute_metrics(actual, predicted)

# --------------------------------------------------------
# ✅ Show Metrics
# --------------------------------------------------------
st.subheader("✅ Model Performance Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.4f}")
col2.metric("MAE", f"{mae:.4f}")
col3.metric("R² Score", f"{r2:.4f}")

# --------------------------------------------------------
# ✅ Explanation of Metrics
# --------------------------------------------------------
st.markdown("""
### 🔎 What These Metrics Mean

✅ **RMSE (Root Mean Squared Error)**  
Measures how far predictions are from actual prices on average.  
- Lower = better  
- Interpreted in the same units as stock price.

✅ **MAE (Mean Absolute Error)**  
Average absolute difference between prediction and real price.  
- Less sensitive to large spikes than RMSE.

✅ **R² Score (Coefficient of Determination)**  
Explains how much of the stock price variation the model captures.  
- 1.0 → Perfect  
- 0.0 → No predictive power  
- Negative → Worse than baseline  

A good stock prediction model typically has:  
- ✅ RMSE **< 5–10**  
- ✅ MAE **low**  
- ✅ R² **0.5+** (your model is around there)
""")

# --------------------------------------------------------
# ✅ Plot 1 — Close Price Over Time
# --------------------------------------------------------
st.subheader("📈 Close Price Over Time")
close_fig = plot_close_price(df)
st.pyplot(close_fig)

# --------------------------------------------------------
# ✅ Plot 2 — Train / Test / Predictions
# --------------------------------------------------------
st.subheader("📉 Train vs Test vs Predictions (LSTM)")
pred_fig = plot_predictions_full(df, actual, predicted, train_len, seq_len)
st.pyplot(pred_fig)

st.subheader("📊 Full Actual Data vs Predicted Data")
fig_full = plot_full_actual_vs_predicted(df, actual, predicted)
st.pyplot(fig_full)

# --------------------------------------------------------
# ✅ Raw Data Preview
# --------------------------------------------------------
with st.expander("📄 View Raw Preprocessed Data"):
    st.dataframe(df)



