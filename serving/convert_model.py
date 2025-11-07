import tensorflow as tf
import os
import sys

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Path to original .keras model
model_dir = "serving/model/e45e81134551430085cf29fe7af76f30/artifacts"
old_keras_path = os.path.join(model_dir, "lstm_stock_model.keras")

# ✅ Paths for output models
h5_path = os.path.join(model_dir, "lstm_stock_model.h5")
new_keras3_path = "models/lstm_stock_model.keras"  # final model used in Docker

os.makedirs("models", exist_ok=True)

print("\n✅ Loading old model:", old_keras_path)
model = tf.keras.models.load_model(old_keras_path)

# ✅ Step 1: Save to H5 (universal format)
print("✅ Saving as H5...")
model.save(h5_path, save_format="h5")

# ✅ Step 2: Reload from H5 (forces new format)
print("✅ Reloading from H5...")
model = tf.keras.models.load_model(h5_path)

# ✅ Step 3: Save final Keras-3 model to /models/
print("✅ Saving final Keras-3 model:", new_keras3_path)
model.save(new_keras3_path, save_format="keras")

print("\n🎉✅ Model fully converted to Keras-3 format!")
print("✅ Use THIS file inside Docker:", new_keras3_path)
