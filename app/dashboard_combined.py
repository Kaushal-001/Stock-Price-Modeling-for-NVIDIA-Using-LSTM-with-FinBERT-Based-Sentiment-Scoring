import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt

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
# ✅ UI Setup
# --------------------------------------------------------
st.set_page_config(
    page_title="NVIDIA — Stock Prediction & Sentiment Dashboard",
    layout="wide"
)

st.title("📊 NVIDIA — Unified Stock Prediction & Sentiment Dashboard")

tab1, tab2 = st.tabs([
    "📈 Stock Prediction (LSTM Model)",
    "📰 News Sentiment Dashboard"
])

# ========================================================
# ✅ TAB 1 — STOCK PREDICTION DASHBOARD
# ========================================================

with tab1:
    st.header("📈 NVIDIA Stock Prediction — LSTM Model")

    MODEL_PATH  = "models/lstm_stock_model.keras"
    SCALER_PATH = "models/standard_scaler.pkl"
    DATA_PATH   = "/Users/kaushaljha/Desktop/Stock_prediction_and_sentiment_analysis/news_data/nvidia_stock.csv"

    # Load & preprocess stock data
    df = load_data(DATA_PATH)
    df = preprocess_stock(df)
    df = df.sort_values("Date").reset_index(drop=True)

    # Load model + scaler
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

    # Run predictions
    actual, predicted, train_len, seq_len = predict_stock(df, model, scaler)

    # Compute Metrics
    rmse, mae, r2 = compute_metrics(actual, predicted)

    # Metrics
    st.subheader("✅ Model Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("R² Score", f"{r2:.4f}")

    st.markdown("""
    ✅ RMSE – error magnitude in USD  
    ✅ MAE – average deviation  
    ✅ R² – variance explained by the model  
    """)

    # Plot 1: Close Price
    st.subheader("📈 Close Price Over Time")
    st.pyplot(plot_close_price(df))

    # Plot 2: Train / Test / Predictions
    st.subheader("📉 Train vs Test vs Predictions (LSTM)")
    st.pyplot(plot_predictions_full(df, actual, predicted, train_len, seq_len))

    # Plot 3: Full Actual vs Predicted
    st.subheader("📊 Full Actual vs Predicted")
    st.pyplot(plot_full_actual_vs_predicted(df, actual, predicted))

    with st.expander("📄 Raw Preprocessed Data"):
        st.dataframe(df)


# ========================================================
# ✅ TAB 2 — SENTIMENT DASHBOARD
# ========================================================

with tab2:

    st.header("📰 NVIDIA News Sentiment Impact on Stock Price")

    # -------------------------------------------------------
    # Load Sentiment Data
    # -------------------------------------------------------
    @st.cache_data
    def load_news_sentiment():
        path = "/Users/kaushaljha/Desktop/Stock_prediction_and_sentiment_analysis/nvidia_news_with_sentiment_score.csv"
        df = pd.read_csv(path)
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df['date'] = df['publishedAt'].dt.date
        return df

    nvidia_news = load_news_sentiment()

    # -------------------------------------------------------
    # Load Latest NVIDIA Stock (30 Days)
    # -------------------------------------------------------
    @st.cache_data
    def load_nvidia_stock():
        start_date = "2025-10-05"
        end_date = "2025-11-04"
        data = yf.download("NVDA", start=start_date, end=end_date, interval="1d")[['Close']]
        data = data.reset_index(drop=False)
        data.columns = ['publishedAt', 'Close']
        data['date'] = pd.to_datetime(data['publishedAt']).dt.date
        return data

    nvidia_stock = load_nvidia_stock()

    # -------------------------------------------------------
    # Merge & Process Sentiment
    # -------------------------------------------------------
    daily_sentiment = (
        nvidia_news.groupby('date')
        .agg({
            'sentiment_score': 'mean',
            'confidence': 'mean',
            'title': 'count'
        })
        .rename(columns={'title': 'article_count'})
        .reset_index()
    )

    merged = pd.merge(nvidia_stock, daily_sentiment, on='date', how='left')
    merged['sentiment_score'].fillna(0, inplace=True)
    merged['article_count'].fillna(0, inplace=True)

    merged['Close_rolling'] = merged['Close'].rolling(3).mean()
    merged['Sentiment_rolling'] = merged['sentiment_score'].rolling(3).mean()

    # -------------------------------------------------------
    # Sentiment Distribution Plot
    # -------------------------------------------------------
    st.subheader("🟢 Sentiment Distribution of News Articles")
    fig_dist = px.histogram(
        nvidia_news, 
        x="sentiment_label", 
        color="sentiment_label",
        title="Sentiment Category Distribution"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # -------------------------------------------------------
    # Sentiment vs Stock Trend
    # -------------------------------------------------------
    st.subheader("📅 Daily Stock Price vs Sentiment Trend")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(merged['date'], merged['Close'], label="Stock Price", color="blue")

    scaled_sentiment = merged['sentiment_score'] * merged['Close'].max()
    ax.plot(merged['date'], scaled_sentiment, '--', color="orange", label="Sentiment (scaled)")

    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # -------------------------------------------------------
    # Correlation Scatter Plot
    # -------------------------------------------------------
    st.subheader("🔗 Sentiment vs Stock Price Correlation")

    corr = merged['Close'].corr(merged['sentiment_score'])
    st.metric("Correlation", f"{corr:.3f}")

    fig_corr = px.scatter(
        merged,
        x="sentiment_score",
        y="Close",
        trendline="ols",
        color="sentiment_score",
        title="Sentiment vs Stock Price"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📄 View Sentiment + Stock Data"):
        st.dataframe(merged)

