# features/sentiment_engineering.py
import pandas as pd
from tqdm import tqdm
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from models.finbert_sentiment import FinBERTSentiment

class SentimentFeatureEngineer:
    """
    Applies FinBERTSentiment model to a DataFrame
    and generates sentiment label, score, and confidence columns.
    """

    def __init__(self):
        self.model = FinBERTSentiment()

    def get_finbert_sentiment(self, text: str) -> pd.Series:
        result = self.model.predict(text)
        label = result["label"]
        score = result["score"]

        # Convert label to numeric value
        if label == "positive":
            sentiment_value = 1
        elif label == "negative":
            sentiment_value = -1
        else:
            sentiment_value = 0

        return pd.Series([label, sentiment_value, score])

    def process_dataset(self, csv_path: str, output_path: str):
        print(f"📂 Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Combine title + description if text column missing
        if "text" not in df.columns:
            df["text"] = (
                df["title"].fillna("") + " " + df["description"].fillna("")
            )

        tqdm.pandas(desc="🔍 Calculating FinBERT Sentiment")
        df[["sentiment_label", "sentiment_score", "confidence"]] = (
            df["text"].progress_apply(self.get_finbert_sentiment)
        )

        df.to_csv(output_path, index=False)
        print(f"✅ Sentiment analysis complete. Saved to {output_path}")
        return df
