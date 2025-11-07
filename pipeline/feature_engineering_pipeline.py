# pipeline.py
import os
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.features.feature_engineering_news import SentimentFeatureEngineer

def main():
    input_csv = "/Users/kaushaljha/Desktop/Stock_prediction_and_sentiment_analysis/news_data/nvidia_financial_news (1).csv"
    output_csv = "nvidia_news_with_sentiment_score.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"❌ Input file not found: {input_csv}")

    engineer = SentimentFeatureEngineer()
    processed_df = engineer.process_dataset(input_csv, output_csv)

    print("📊 Sample of processed data:")
    print(processed_df.head())


if __name__ == "__main__":
    main()
