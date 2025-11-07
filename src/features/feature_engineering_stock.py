# src/features/feature_engineering_stock.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class StockFeatureEngineer:
    def __init__(self, seq_len=60):
        """
        Handles feature scaling and sequence creation for LSTM models.
        Uses StandardScaler for normalization (mean=0, std=1).
        """
        self.seq_len = seq_len
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame, column="Close"):
        """
        Fit the scaler on stock prices and generate LSTM sequences.
        Returns scaled features (X) and targets (y).
        """
        scaled_data = self.scaler.fit_transform(df[[column]].values)
        X, y = self._create_sequences(scaled_data)
        return np.array(X), np.array(y)

    def transform(self, df: pd.DataFrame, column="Close"):
        """
        Transform new data using the previously fitted scaler.
        """
        scaled_data = self.scaler.transform(df[[column]].values)
        X, y = self._create_sequences(scaled_data)
        return np.array(X), np.array(y)

    def _create_sequences(self, data):
        """
        Internal helper to create (X, y) sequences for LSTM input.
        X shape: (samples, seq_len, 1)
        y shape: (samples,)
        """
        X, y = [], []
        for i in range(self.seq_len, len(data)):
            X.append(data[i - self.seq_len:i, 0])
            y.append(data[i, 0])
        return np.reshape(X, (len(X), self.seq_len, 1)), np.array(y)

    def save_scaler(self, path="models/scaler.pkl"):
        """Save the fitted scaler for later inference."""
        import joblib, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path="models/scaler.pkl"):
        """Load a previously saved scaler."""
        import joblib
        self.scaler = joblib.load(path)
