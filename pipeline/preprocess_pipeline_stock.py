import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from src.data.preprocess.preprocess_stock import preprocess_stock
from src.data.load_data import load_data

def main():
    stock_path = os.path.join("news_data", "nvidia_stock.csv")
    
    # Load and preprocess
    stock_df = load_data(stock_path)
    processed_stock = preprocess_stock(stock_df)

    print("✅ Preprocessing Completed for Stock Data!")
    print("Shape Before:", stock_df.shape)
    print("Shape After:", processed_stock.shape)
    print(processed_stock.head())

if __name__ == "__main__":
    main()