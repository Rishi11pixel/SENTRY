import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "tinyml"

KERAS_MODEL = MODEL_DIR / "sentry_tinyml.keras"
TFLITE_MODEL = MODEL_DIR / "sentry_tinyml_int8.tflite"

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "sentry_synthetic_dataset.csv"
)

# ============================================================
# Features
# ============================================================

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0567",
    "dVdt_max",
    "temperature",
    "humidity",
]

# ============================================================
# Load model
# ============================================================

print("Loading Keras model...")

model = tf.keras.models.load_model(KERAS_MODEL)

# ============================================================
# Load scaler parameters
# ============================================================

mean = np.load(
    MODEL_DIR / "scaler_mean.npy"
)

scale = np.load(
    MODEL_DIR / "scaler_scale.npy"
)

# ============================================================
# Load dataset for representative calibration
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].values.astype(np.float32)

# Apply the EXACT same preprocessing used during training.
X = (X - mean) / scale

# ============================================================
# Representative dataset
# ============================================================

def representative_dataset():

    for sample in X:

        sample = sample.reshape(1, 7).astype(np.float32)

        yield [sample]


# ============================================================
# Convert to INT8
# ============================================================

print("Converting to full INT8 TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# ============================================================
# Save
# ============================================================

with open(TFLITE_MODEL, "wb") as f:
    f.write(tflite_model)

print()
print("=" * 70)
print("TINYML CONVERSION COMPLETE")
print("=" * 70)

print(f"\nModel: {TFLITE_MODEL}")

print(
    f"Size: {len(tflite_model):,} bytes "
    f"({len(tflite_model) / 1024:.2f} KB)"
)