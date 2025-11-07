import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm  # Needed for trendline
import numpy as np
import os

st.set_page_config(page_title="NVIDIA Stock & Sentiment Dashboard", layout="wide")

# -----------------------------------------------------------
# 1️⃣ Load News Sentiment Data
# -----------------------------------------------------------
@st.cache_data
def load_news_sentiment():
    path = os.path.join("news_data", "nvidia_news_with_sentiment_score.csv")
    df = pd.read_csv(path)

    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['date'] = df['publishedAt'].dt.date
    return df

nvidia_news = load_news_sentiment()

# -----------------------------------------------------------
# 2️⃣ Fetch Latest NVIDIA Stock Data
# -----------------------------------------------------------
@st.cache_data
def load_nvidia_stock():
    start_date = "2025-10-05"
    end_date = "2025-11-04"
    
    # Download NVDA stock data
    nvidia_stock = yf.download("NVDA", start=start_date, end=end_date, interval="1d")[['Close']]
    nvidia_stock = nvidia_stock.reset_index(drop=False)
    nvidia_stock.columns = ['publishedAt', 'Close']
    nvidia_stock['date'] = pd.to_datetime(nvidia_stock['publishedAt']).dt.date
    return nvidia_stock

nvidia_stock = load_nvidia_stock()

# -----------------------------------------------------------
# 3️⃣ Aggregate Daily Sentiment and Merge with Stock
# -----------------------------------------------------------
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
merged['confidence'].fillna(0, inplace=True)
merged['article_count'].fillna(0, inplace=True)

# -----------------------------------------------------------
# 4️⃣ Rolling Averages (optional smoothing)
# -----------------------------------------------------------
merged['Close_rolling'] = merged['Close'].rolling(window=3).mean()
merged['Sentiment_rolling'] = merged['sentiment_score'].rolling(window=3).mean()

# -----------------------------------------------------------
# 5️⃣ Dashboard Title
# -----------------------------------------------------------
st.title("📊 NVIDIA News Sentiment vs Stock Price Dashboard")
st.markdown(
    """
    Explore how **FinBERT-based sentiment** from financial news correlates with 
    **NVIDIA's daily stock performance**.  
    _Note: News sentiment data is limited to the last 30 days, which may not be sufficient 
    to establish a strong statistical correlation._
    """
)

# -----------------------------------------------------------
# 6️⃣ Sentiment Distribution
# -----------------------------------------------------------
st.subheader("🟢 Sentiment Distribution of News Articles")
fig_dist = px.histogram(
    nvidia_news, 
    x="sentiment_label", 
    color="sentiment_label",
    title="Sentiment Category Distribution",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_dist, use_container_width=True)

# -----------------------------------------------------------
# 7️⃣ Stock Price vs Sentiment Trend (Matplotlib Style)
# -----------------------------------------------------------
st.subheader("📅 NVIDIA Stock Price vs Sentiment Trend (30-Day Overview)")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(merged['date'], merged['Close'], label='Stock Closing Price (USD)', color='blue', linewidth=2)

# Scale sentiment for visual comparison
scaled_sentiment = merged['sentiment_score'] * merged['Close'].max()
ax.plot(merged['date'], scaled_sentiment, label='Sentiment Score (scaled)', color='orange', linestyle='--', linewidth=2)

ax.set_title("NVIDIA Stock Price vs News Sentiment Trend", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Value (Stock Price & Scaled Sentiment)", fontsize=12)
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.markdown(
    """
    🔍 **Interpretation:**  
    The **blue line** represents NVIDIA’s daily stock closing price, while the **orange dashed line** shows
    the **FinBERT-based sentiment score**, scaled to the same range for visual comparison.  
    While some peaks and dips may align, note that:
    - The dataset covers **only 30 days of news**, so correlations may appear coincidental.
    - Broader patterns would need **longer-term data** for statistical reliability.
    """
)

# -----------------------------------------------------------
# 8️⃣ Sentiment vs Stock Price Correlation — Enhanced
# -----------------------------------------------------------
st.subheader("🔗 Sentiment vs Stock Price Correlation")

# Calculate correlation coefficient
corr = merged['Close'].corr(merged['sentiment_score'])
st.markdown(f"### 📈 Correlation Coefficient: `{corr:.3f}`")

# Create scatter plot with regression trendline
fig_corr = px.scatter(
    merged,
    x="sentiment_score",
    y="Close",
    trendline="ols",
    color="sentiment_score",
    color_continuous_scale="RdYlGn",
    title="Relationship Between News Sentiment and NVIDIA Stock Price",
    labels={
        "sentiment_score": "Daily Average Sentiment Score",
        "Close": "NVIDIA Closing Price (USD)"
    }
)
st.plotly_chart(fig_corr, use_container_width=True)

# Generate interpretation text dynamically
if corr > 0.3:
    explanation = (
        "📊 **Positive Correlation Detected:** As sentiment becomes more positive, "
        "NVIDIA’s stock price tends to rise — though given the 30-day limit, this should be interpreted cautiously."
    )
elif corr < -0.3:
    explanation = (
        "📉 **Negative Correlation Detected:** Negative sentiment may accompany lower stock prices. "
        "However, with limited data, this could be a short-term pattern."
    )
else:
    explanation = (
        "⚖️ **Weak or No Correlation:** Over the 30-day period, sentiment and price show little alignment. "
        "This is expected for short timeframes where market factors outweigh news tone."
    )

st.markdown(explanation)

# -----------------------------------------------------------
# 💾 Optional: Download Processed Data
# -----------------------------------------------------------
st.download_button(
    label="⬇️ Download Merged Data as CSV",
    data=merged.to_csv(index=False).encode("utf-8"),
    file_name="nvidia_sentiment_stock_merged.csv",
    mime="text/csv"
)

"""
to run this app use the command below

streamlit run app/ui_sentiment.py
"""