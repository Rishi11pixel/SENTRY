import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models" / "tinyml"

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
# Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].values.astype(np.float32)
y_text = df["label"].values

encoder = LabelEncoder()

y = encoder.fit_transform(y_text)


# ============================================================
# Same test split
# ============================================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ============================================================
# Load scaler
# ============================================================

mean = np.load(
    MODEL_DIR / "scaler_mean.npy"
)

scale = np.load(
    MODEL_DIR / "scaler_scale.npy"
)


X_test = (X_test - mean) / scale

X_test = X_test.astype(np.float32)


# ============================================================
# Load TFLite model
# ============================================================

print("Loading INT8 model...")

interpreter = tf.lite.Interpreter(
    model_path=str(TFLITE_MODEL)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)


# ============================================================
# Quantization parameters
# ============================================================

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)


print("\nInput quantization:")
print(f"scale      = {input_scale}")
print(f"zero point = {input_zero_point}")

print("\nOutput quantization:")
print(f"scale      = {output_scale}")
print(f"zero point = {output_zero_point}")


# ============================================================
# Run inference
# ============================================================

predictions = []

confidences = []


for sample in X_test:

    sample = sample.reshape(1, 7)

    # Float → INT8
    quantized_input = (
        sample / input_scale
        + input_zero_point
    )

    quantized_input = np.round(
        quantized_input
    ).astype(np.int8)

    interpreter.set_tensor(
        input_details[0]["index"],
        quantized_input,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # INT8 → approximate probability
    probabilities = (
        output.astype(np.float32)
        - output_zero_point
    ) * output_scale

    probabilities = probabilities[0]

    prediction = int(
        np.argmax(probabilities)
    )

    predictions.append(prediction)

    confidences.append(
        float(np.max(probabilities))
    )


# ============================================================
# Results
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions,
)

print("\n" + "=" * 70)
print("INT8 TINYML RESULTS")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
    )
)

print(
    f"\nAverage confidence: "
    f"{np.mean(confidences):.4f}"
)