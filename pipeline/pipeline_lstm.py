# pipeline/pipeline_lstm.py

import os, sys
import mlflow
import mlflow.tensorflow
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from src.data.load_data import load_data
from src.data.preprocess.preprocess_stock import preprocess_stock
from src.features.feature_engineering_stock import StockFeatureEngineer
from src.models.lstm_model import build_lstm_model


def main():

    print("\n🚀 Starting LSTM Training Pipeline...\n")

    DATA_PATH = os.path.join("news_data", "nvidia_stock.csv")

    # -------------------------------------------------------------
    # ✅ STEP 1 — Load Data
    # -------------------------------------------------------------
    print("📥 Loading CSV data...")
    df = load_data(DATA_PATH)
    print(f"✅ Raw data loaded → Shape: {df.shape}")
    print(df.head(), "\n")

    # -------------------------------------------------------------
    # ✅ STEP 2 — Preprocess
    # -------------------------------------------------------------
    print("🛠️ Preprocessing data...")
    df = preprocess_stock(df)
    print("✅ Preprocessing complete!")
    print("🔍 Columns:", df.columns.tolist())
    print(df.head(), "\n")

    # -------------------------------------------------------------
    # ✅ STEP 3 — Extract Close Prices
    # -------------------------------------------------------------
    print("📊 Extracting 'Close' column...")
    close_data = df[["Close"]].values
    print("✅ Close data extracted → Shape:", close_data.shape)
    print("   Min:", close_data.min(), "Max:", close_data.max(), "\n")

    # -------------------------------------------------------------
    # ✅ STEP 4 — Train/Test Split
    # -------------------------------------------------------------
    print("✂️ Splitting dataset (90% train, 10% test)...")
    split_ratio = 0.9
    train_len = int(len(close_data) * split_ratio)

    train_data = close_data[:train_len]
    test_data = close_data[train_len:]

    print(f"✅ Train length: {len(train_data)}")
    print(f"✅ Test length:  {len(test_data)}\n")

    # -------------------------------------------------------------
    # ✅ STEP 5 — Fit Scaler ONLY on Train
    # -------------------------------------------------------------
    SEQ_LEN = 60
    engineer = StockFeatureEngineer(seq_len=SEQ_LEN)

    print("🔧 Fitting StandardScaler on train data...")
    engineer.scaler.fit(train_data)
    print("✅ Scaler fitted!")
    print("   Mean:", engineer.scaler.mean_)
    print("   Var :", engineer.scaler.var_, "\n")

    scaled_train = engineer.scaler.transform(train_data)
    scaled_test = engineer.scaler.transform(test_data)

    print("✅ Scaled train shape:", scaled_train.shape)
    print("✅ Scaled test  shape:", scaled_test.shape, "\n")

    # -------------------------------------------------------------
    # ✅ STEP 6 — Create Sequences
    # -------------------------------------------------------------
    print("🧩 Creating sequences (X, y)...")

    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(scaled_train, SEQ_LEN)
    X_test, y_test = create_sequences(scaled_test, SEQ_LEN)

    print("✅ X_train:", X_train.shape)
    print("✅ y_train:", y_train.shape)
    print("✅ X_test :", X_test.shape)
    print("✅ y_test :", y_test.shape, "\n")

    # -------------------------------------------------------------
    # ✅ STEP 7 — Build Model
    # -------------------------------------------------------------
    print("🏗️ Building LSTM model...")
    model = build_lstm_model(
        input_shape=(SEQ_LEN, 1),
        lstm_units=128,
        dense_unit=128,
        dropout_rate=0.5350809294827892,
        optimizer="rmsprop"
    )
    print("✅ Model built!\n")

    # -------------------------------------------------------------
    # ✅ STEP 8 — Setup MLflow
    # -------------------------------------------------------------
    print("📡 Setting MLflow tracking...")
    mlflow.set_tracking_uri("file:///Users/kaushaljha/Desktop/Stock_prediction_and_sentiment_analysis/mlruns")
    mlflow.set_experiment("nvidia_lstm_stock_prediction")
    print("✅ MLflow Ready!\n")

    # -------------------------------------------------------------
    # ✅ STEP 9 — Train Model
    # -------------------------------------------------------------
    print("🏋️ Starting training...")

    with mlflow.start_run(run_name="lstm_final_run"):

        mlflow.log_param("sequence_length", SEQ_LEN)
        mlflow.log_param("optimizer", "rmsprop")
        mlflow.log_param("lstm_units", 128)
        mlflow.log_param("dense_units", 128)

        early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=14,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )

        print("\n✅ Training complete!")

        # ---------------------------------------------------------
        # ✅ STEP 10 — Evaluate with All Metrics
        # ---------------------------------------------------------
        print("📊 Computing evaluation metrics...")

        # Model's final test predictions
        y_pred_scaled = model.predict(X_test)

        # Undo standard scaling
        y_pred = engineer.scaler.inverse_transform(y_pred_scaled)
        y_true = engineer.scaler.inverse_transform(y_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"✅ RMSE: {rmse:.4f}")
        print(f"✅ MAE : {mae:.4f}")
        print(f"✅ R²  : {r2:.4f}\n")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)

        # ---------------------------------------------------------
        # ✅ STEP 11 — Save Model + Scaler
        # ---------------------------------------------------------
        print("💾 Saving model and scaler...")

        os.makedirs("models", exist_ok=True)
        MODEL_PATH = "models/lstm_stock_model.keras"
        SCALER_PATH = "models/standard_scaler.pkl"

        model.save(MODEL_PATH)
        joblib.dump(engineer.scaler, SCALER_PATH)

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(SCALER_PATH)

        print(f"✅ Model saved → {MODEL_PATH}")
        print(f"✅ Scaler saved → {SCALER_PATH}")
        print("✅ MLflow Run Complete!\n")

    print("🎉 Pipeline Finished Successfully!")


if __name__ == "__main__":
    main()
