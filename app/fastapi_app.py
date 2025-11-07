from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import pandas as pd
import os, sys
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from serving.inference import load_model_and_scaler
from src.data.load_data import load_data
from src.data.preprocess.preprocess_stock import preprocess_stock


app = FastAPI(title="Stock Prediction + Sentiment API")

# ✅ Allow frontend (Streamlit) to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Paths
# Model files (copied to /app/models)
MODEL_PATH = os.path.join("models", "lstm_stock_model.keras")
SCALER_PATH = os.path.join("models", "standard_scaler.pkl")
# Stock CSV
STOCK_CSV = os.path.join("news_data", "nvidia_stock.csv")
# Sentiment CSV
SENTIMENT_CSV = os.path.join("news_data", "nvidia_news_with_sentiment_score.csv")


# ✅ Load model + scaler once
df_stock = load_data(STOCK_CSV)
df_stock = preprocess_stock(df_stock)

model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)


@app.get("/")
def root():
    return {"status": "ok"}


# ✅ ✅ NEW: Redirect to Streamlit Stock Dashboard
@app.get("/predict-stock")
def redirect_stock_dashboard():
    return RedirectResponse(url="http://localhost:8503")


# ✅ ✅ NEW: Redirect to Streamlit Sentiment Dashboard
@app.get("/sentiment")
def redirect_sentiment_dashboard():
    return RedirectResponse(url="http://localhost:8502")




"""
to run this app use the command below

streamlit run app/dashboard_stock.py --server.port 8503

streamlit run app/ui_sentiment.py --server.port 8502

uvicorn api.fastapi_app:app --reload --port 8000

"""