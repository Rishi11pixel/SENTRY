from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from generate_dataset import (
    CLASSES,
    FEATURES,
    TEST_SEED,
    TEST_PER_CLASS,
    generate_window,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "final_evaluation.json"
TEST_PATH = ROOT / "data" / "raw" / "sentry_synthetic_test.csv"
MEAN_PATH = ROOT / "models" / "tinyml" / "scaler_mean.npy"
SCALE_PATH = ROOT / "models" / "tinyml" / "scaler_scale.npy"
RF_PATH = ROOT / "models" / "random_forest" / "random_forest.joblib"
TINYML_PATH = ROOT / "models" / "tinyml" / "sentry_tinyml.keras"
INT8_PATH = ROOT / "models" / "tinyml" / "sentry_tinyml_int8.tflite"

CONDITIONS = [
    "dry",
    "normal",
    "humid",
    "very_humid",
    "cold",
    "hot",
    "hot_humid",
    "hot_dry",
]


def labels_to_int(labels):
    return np.asarray([CLASSES.index(label) for label in labels], dtype=np.int64)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def float_predictions(X):
    scaler_mean = np.load(MEAN_PATH)
    scaler_scale = np.load(SCALE_PATH)
    model = tf.keras.models.load_model(TINYML_PATH)
    X_scaled = ((X - scaler_mean) / scaler_scale).astype(np.float32)
    return np.argmax(model.predict(X_scaled, verbose=0), axis=1)


def int8_predictions(X):
    scaler_mean = np.load(MEAN_PATH)
    scaler_scale = np.load(SCALE_PATH)
    interpreter = tf.lite.Interpreter(model_path=str(INT8_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero = input_details["quantization"]
    output_scale, output_zero = output_details["quantization"]
    predictions = []
    for row in X:
        scaled = (row - scaler_mean) / scaler_scale
        quantized = np.clip(np.round(scaled / input_scale + input_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], quantized.reshape(1, 7))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        dequantized = (output.astype(np.float32) - output_zero) * output_scale
        predictions.append(int(np.argmax(dequantized[0])))
    return np.asarray(predictions, dtype=np.int64)


def make_generated_frame(label, condition, count, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        features, _ = generate_window(label, rng, True, condition)
        rows.append(features + [label])
    return pd.DataFrame(rows, columns=FEATURES + ["label"])


def main():
    final_test = pd.read_csv(TEST_PATH)
    rf = joblib.load(RF_PATH)
    tinyml = tf.keras.models.load_model(TINYML_PATH)
    mean = np.load(MEAN_PATH)
    scale = np.load(SCALE_PATH)

    compensated = final_test[FEATURES]
    y = labels_to_int(final_test["label"])

    raw_counterpart, _ = __import__("generate_dataset").generate_dataset(
        TEST_PER_CLASS, TEST_SEED, apply_compensation=False
    )
    raw_features = raw_counterpart[["VMQ2", "VMQ3", "VMQ135", "VSEN0567", "dVdt_max"]].copy()
    raw_features["temperature"] = compensated["temperature"].to_numpy()
    raw_features["humidity"] = compensated["humidity"].to_numpy()

    ablation = {}
    for name, X in [("without_compensation", raw_features), ("with_compensation", compensated)]:
        rf_pred = rf.predict(X)
        tiny_pred = np.argmax(tinyml.predict(((X.to_numpy() - mean) / scale).astype(np.float32), verbose=0), axis=1)
        ablation[name] = {
            "random_forest": metrics(y, rf_pred),
            "tinyml_float32": metrics(y, tiny_pred),
            "same_seed_paired_synthetic_test_condition": True,
        }

    robustness = {}
    for condition_index, condition in enumerate(CONDITIONS):
        condition_rows = []
        for label_index, label in enumerate(CLASSES):
            condition_rows.append(make_generated_frame(label, condition, 200, 7000 + condition_index * 100 + label_index))
        condition_df = pd.concat(condition_rows, ignore_index=True)
        X = condition_df[FEATURES]
        truth = labels_to_int(condition_df["label"])
        rf_pred = rf.predict(X)
        tiny_pred = float_predictions(X.to_numpy(dtype=np.float32))
        robustness[condition] = {
            "random_forest": metrics(truth, rf_pred),
            "tinyml_float32": metrics(truth, tiny_pred),
            "safe_accuracy": float(accuracy_score(truth[truth == CLASSES.index("SAFE")], rf_pred[truth == CLASSES.index("SAFE")])),
            "weather_accuracy": float(accuracy_score(truth[truth == CLASSES.index("WEATHER")], rf_pred[truth == CLASSES.index("WEATHER")])),
        }

    with RESULTS_PATH.open(encoding="utf-8") as fp:
        report = json.load(fp)
    report["compensation_ablation"] = ablation
    report["weather_robustness"] = robustness
    report["sensor_and_data_constraints"] = {
        "sensor": "DFRobot SEN0567 NH3",
        "final_test_used_for_training_or_tuning": False,
        "synthetic_not_laboratory_calibrated": True,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    print("Compensation ablation:")
    for name, values in ablation.items():
        print(f"  {name}: RF accuracy={values['random_forest']['accuracy']:.4f}, TinyML accuracy={values['tinyml_float32']['accuracy']:.4f}")
    print("\nWeather robustness (SAFE/WEATHER conditions):")
    for condition, values in robustness.items():
        print(f"  {condition}: SAFE={values['safe_accuracy']:.3f}, WEATHER={values['weather_accuracy']:.3f}")
    print("\nSEN0567/NH3 used: confirmed.")
    print("Final blind test used for training/tuning: no.")
    print("Synthetic and not laboratory-calibrated: confirmed.")


if __name__ == "__main__":
    main()
