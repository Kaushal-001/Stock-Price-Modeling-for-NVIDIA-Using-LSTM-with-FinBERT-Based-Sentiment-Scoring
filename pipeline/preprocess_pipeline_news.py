import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from src.data.preprocess.preprocess_news import preprocess_news
from src.data.load_data import load_data

def main():
    news_path = "/Users/kaushaljha/Desktop/Stock_prediction_and_sentiment_analysis/news_data/nvidia_financial_news (1).csv"
    
    # Load and preprocess
    news_df = load_data(news_path)
    processed_news = preprocess_news(news_df)

    print("✅ Preprocessing Completed for News Data!")
    print("Shape Before:", news_df.shape)
    print("Shape After:", processed_news.shape)
    print(processed_news.head())

if __name__ == "__main__":
    main()