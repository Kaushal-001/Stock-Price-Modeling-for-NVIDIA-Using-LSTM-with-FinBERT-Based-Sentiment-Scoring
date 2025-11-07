import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.load_data import load_data

def main():
    stock_path = "/Users/kaushaljha/Desktop/Stock_prediction_and_sentiment_analysis/news_data/nvidia_stock.csv"
    news_path = "/Users/kaushaljha/Desktop/Stock_prediction_and_sentiment_analysis/news_data/nvidia_financial_news (1).csv"
    
    # Load both datasets
    stock_df = load_data(stock_path)
    news_df = load_data(news_path)

    print("✅ Stock data shape:", stock_df.shape)
    print("First few column of stock data\n",stock_df.head())
    print("✅ News data shape:", news_df.shape)
    print("Data Loaded Successfully!")

if __name__ == "__main__":
    main()


"""
to run this pipeline to test whether the data is being loaded or not run the below command:

python pipeline/load_data_pipeline.py

"""