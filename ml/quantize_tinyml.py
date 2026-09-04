from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATA_PATH = ROOT / "data" / "raw" / "sentry_synthetic_dataset.csv"
MODEL_PATH = ROOT / "models" / "tinyml" / "sentry_tinyml.keras"
SCALER_MEAN_PATH = ROOT / "models" / "tinyml" / "scaler_mean.npy"
SCALER_SCALE_PATH = ROOT / "models" / "tinyml" / "scaler_scale.npy"
OUTPUT_MODEL_PATH = ROOT / "models" / "tinyml" / "sentry_tinyml_int8.tflite"
METADATA_PATH = ROOT / "results" / "tinyml_int8_metadata.json"

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0567",
    "dVdt_max",
    "temperature",
    "humidity",
]

CLASSES = [
    "SAFE",
    "WEATHER",
    "ALCOHOL",
    "EXPLOSIVE",
    "NARCOTIC",
]


def load_training_dataframe() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_DATA_PATH)

    if len(df) != 60000:
        raise ValueError(f"Expected 60,000 training samples, found {len(df)} rows.")

    return df


def representative_dataset():
    df = load_training_dataframe()
    subset = df.sample(n=500, random_state=42).reset_index(drop=True)
    X = subset[FEATURES].to_numpy(dtype=np.float32)
    scaler_mean = np.load(SCALER_MEAN_PATH)
    scaler_scale = np.load(SCALER_SCALE_PATH)
    X_scaled = (X - scaler_mean) / scaler_scale
    X_scaled = X_scaled.astype(np.float32)

    for sample in X_scaled:
        yield [sample.astype(np.float32)]


def print_tensor_details(interpreter: tf.lite.Interpreter, tensor_name: str) -> tuple[str, float, int]:
    tensor_index = interpreter.get_input_details()[0]["index"] if tensor_name == "input" else interpreter.get_output_details()[0]["index"]
    details = interpreter.get_input_details()[0] if tensor_name == "input" else interpreter.get_output_details()[0]

    dtype = details["dtype"]
    scale = details.get("quantization_parameters", {}).get("scale", 0.0)
    zero_point = details.get("quantization_parameters", {}).get("zero_point", 0)
    return str(dtype), float(scale), int(zero_point)


def main() -> None:
    model = tf.keras.models.load_model(MODEL_PATH)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL_PATH.write_bytes(tflite_model)

    model_size_bytes = OUTPUT_MODEL_PATH.stat().st_size
    model_size_kb = model_size_bytes / 1024.0

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = tuple(input_details["shape"])
    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]

    input_quantization = input_details.get("quantization", (0.0, 0))
    output_quantization = output_details.get("quantization", (0.0, 0))
    input_scale = float(input_quantization[0])
    input_zero_point = int(input_quantization[1])
    output_scale = float(output_quantization[0])
    output_zero_point = int(output_quantization[1])

    if input_dtype != np.int8:
        raise ValueError(f"Model input dtype is not np.int8: {input_dtype}")

    if output_dtype != np.int8:
        raise ValueError(f"Model output dtype is not np.int8: {output_dtype}")

    metadata = {
        "model_path": str(OUTPUT_MODEL_PATH.relative_to(ROOT)),
        "model_size_bytes": int(model_size_bytes),
        "model_size_kb": float(model_size_kb),
        "input_shape": list(map(int, input_shape)),
        "input_dtype": str(input_dtype),
        "input_quantization_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_dtype": str(output_dtype),
        "output_quantization_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "representative_sample_count": int(500),
        "representative_dataset_source": str(TRAIN_DATA_PATH.relative_to(ROOT)),
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Output model: {OUTPUT_MODEL_PATH}")
    print(f"Model size bytes: {model_size_bytes}")
    print(f"Model size KB: {model_size_kb:.2f}")
    print(f"Input tensor shape: {input_shape}")
    print(f"Input dtype: {input_dtype}")
    print(f"Input quantization scale: {input_scale}")
    print(f"Input zero point: {input_zero_point}")
    print(f"Output dtype: {output_dtype}")
    print(f"Output quantization scale: {output_scale}")
    print(f"Output zero point: {output_zero_point}")

    print("INT8 quantization completed.")
    print("Representative data came only from the 60,000-sample training dataset.")
    print("Final 12,000-sample holdout was NOT used.")

    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
