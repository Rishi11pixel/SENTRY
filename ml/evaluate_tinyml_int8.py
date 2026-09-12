from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent.parent

TEST_DATA_PATH = ROOT / "data" / "raw" / "sentry_synthetic_test.csv"
MODEL_PATH = ROOT / "models" / "tinyml" / "sentry_tinyml_int8.tflite"
SCALER_MEAN_PATH = ROOT / "models" / "tinyml" / "scaler_mean.npy"
SCALER_SCALE_PATH = ROOT / "models" / "tinyml" / "scaler_scale.npy"
RESULT_DIR = ROOT / "results"
JSON_PATH = RESULT_DIR / "final_int8_evaluation.json"
PLOT_PATH = RESULT_DIR / "final_int8_confusion_matrix.png"
FLOAT32_REPORT_PATH = RESULT_DIR / "final_evaluation.json"

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

THREAT_INDICES = np.array([CLASSES.index("EXPLOSIVE"), CLASSES.index("NARCOTIC")])


def compute_threat_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_threat = np.isin(y_true, THREAT_INDICES)
    predicted_threat = np.isin(y_pred, THREAT_INDICES)
    tn, fp, fn, tp = confusion_matrix(
        true_threat,
        predicted_threat,
        labels=[False, True],
    ).ravel()
    threat_recall = tp / (tp + fn) if tp + fn else 0.0
    threat_fnr = fn / (tp + fn) if tp + fn else 0.0
    threat_precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threat_definition": "EXPLOSIVE + NARCOTIC",
        "non_threat_definition": "SAFE + WEATHER + ALCOHOL",
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "threat_recall": float(threat_recall),
        "threat_false_negative_rate": float(threat_fnr),
        "threat_precision": float(threat_precision),
    }


def load_test_dataframe() -> pd.DataFrame:
    df = pd.read_csv(TEST_DATA_PATH)
    if len(df) != 12000:
        raise ValueError(f"Expected exactly 12,000 rows in final test set, found {len(df)}.")

    actual_distribution = df["label"].value_counts().reindex(CLASSES, fill_value=0)
    expected_distribution = pd.Series({label: 2400 for label in CLASSES})
    if not actual_distribution.equals(expected_distribution):
        raise ValueError(
            "Final test set class distribution is not exactly 2,400 samples per class. "
            f"Actual distribution: {actual_distribution.to_dict()}"
        )

    return df


def load_model_details() -> tuple[tf.lite.Interpreter, dict, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    input_shape = tuple(input_details["shape"])
    output_shape = tuple(output_details["shape"])

    if input_dtype != np.int8:
        raise ValueError(f"Model input dtype is not np.int8: {input_dtype}")
    if output_dtype != np.int8:
        raise ValueError(f"Model output dtype is not np.int8: {output_dtype}")

    input_scale = float(input_details.get("quantization", (0.0, 0))[0])
    input_zero_point = int(input_details.get("quantization", (0.0, 0))[1])
    output_scale = float(output_details.get("quantization", (0.0, 0))[0])
    output_zero_point = int(output_details.get("quantization", (0.0, 0))[1])

    if input_scale == 0:
        raise ValueError("Input quantization scale is zero; model is invalid for INT8 inference.")
    if output_scale == 0:
        raise ValueError("Output quantization scale is zero; model is invalid for INT8 inference.")

    if input_shape[-1] != len(FEATURES):
        raise ValueError(
            f"Input shape {input_shape} is not compatible with the expected {len(FEATURES)} features."
        )

    return interpreter, {
        "input_dtype": input_dtype,
        "input_shape": input_shape,
        "input_scale": input_scale,
        "input_zero_point": input_zero_point,
    }, {
        "output_dtype": output_dtype,
        "output_shape": output_shape,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
        zero_division=0,
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))

    per_class = {}
    for idx, label in enumerate(CLASSES):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
        "threat_safety": compute_threat_metrics(y_true, y_pred),
    }


def plot_confusion_matrix(cm: np.ndarray) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("Blues")
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.set_title("TinyML INT8 — Final Test Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            cell_color = cmap(value / max(1, cm.max()))
            r, g, b, _ = cell_color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                weight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_test_dataframe()
    X_raw = df[FEATURES].to_numpy(dtype=np.float32)
    y_true = df["label"].to_numpy()
    y_true_encoded = np.array([CLASSES.index(label) for label in y_true], dtype=np.int64)

    interp, input_meta, output_meta = load_model_details()
    scaler_mean = np.load(SCALER_MEAN_PATH)
    scaler_scale = np.load(SCALER_SCALE_PATH)

    input_scale = input_meta["input_scale"]
    input_zero_point = input_meta["input_zero_point"]
    output_scale = output_meta["output_scale"]
    output_zero_point = output_meta["output_zero_point"]

    predictions = []
    for sample in X_raw:
        x_scaled = (sample - scaler_mean) / scaler_scale
        x_int8 = np.round(x_scaled / input_scale + input_zero_point).astype(np.int8)
        x_int8 = np.clip(x_int8, -128, 127)

        interp.set_tensor(interp.get_input_details()[0]["index"], np.expand_dims(x_int8, axis=0).astype(np.int8))
        interp.invoke()

        output_int8 = interp.get_tensor(interp.get_output_details()[0]["index"])
        output_float = (output_int8.astype(np.float32) - output_zero_point) * output_scale
        prediction = int(np.argmax(output_float[0]))
        predictions.append(prediction)

    predictions = np.asarray(predictions, dtype=np.int64)
    metrics = compute_classification_metrics(y_true_encoded, predictions)

    model_size_bytes = MODEL_PATH.stat().st_size
    model_size_kb = model_size_bytes / 1024.0

    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"], dtype=np.int64))

    report = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "dataset_path": str(TEST_DATA_PATH.relative_to(ROOT)),
        "rows": int(len(df)),
        "class_distribution": {label: int(df["label"].value_counts().get(label, 0)) for label in CLASSES},
        "features": FEATURES,
        "classes": CLASSES,
        "input_dtype": str(input_meta["input_dtype"]),
        "input_shape": list(map(int, input_meta["input_shape"])),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_dtype": str(output_meta["output_dtype"]),
        "output_shape": list(map(int, output_meta["output_shape"])),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "model_size_bytes": int(model_size_bytes),
        "model_size_kb": float(model_size_kb),
        "metrics": metrics,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    with FLOAT32_REPORT_PATH.open(encoding="utf-8") as fp:
        float32_report = json.load(fp)["tinyml_float32"]

    float32_accuracy = float(float32_report["accuracy"])
    float32_macro_f1 = float(float32_report["macro_f1"])
    int8_accuracy = metrics["accuracy"] * 100.0
    int8_macro_f1 = metrics["macro_f1"] * 100.0
    accuracy_delta = int8_accuracy - float32_accuracy * 100.0
    macro_f1_delta = int8_macro_f1 - float32_macro_f1 * 100.0

    print("TinyML Float32:")
    print(f"Accuracy: {float32_accuracy * 100:.2f}%")
    print(f"Macro F1: {float32_macro_f1 * 100:.2f}%")
    print()
    print("TinyML INT8:")
    print(f"Accuracy: {int8_accuracy:.2f}%")
    print(f"Macro F1: {int8_macro_f1:.2f}%")
    print()
    print(f"INT8 accuracy change: {accuracy_delta:.2f} percentage points")
    print(f"INT8 macro F1 change: {macro_f1_delta:.2f} percentage points")
    print()
    print(f"Input scale: {input_scale}")
    print(f"Input zero point: {input_zero_point}")
    print(f"Output scale: {output_scale}")
    print(f"Output zero point: {output_zero_point}")
    print(f"Model size bytes: {model_size_bytes}")
    print(f"Model size KB: {model_size_kb:.2f}")
    print(f"Saved evaluation JSON: {JSON_PATH}")
    print(f"Saved confusion matrix plot: {PLOT_PATH}")
    print("FINAL INT8 BLIND EVALUATION COMPLETE.")
    print("Final 12,000-sample holdout was used only for evaluation.")
    print("INT8 model was not modified.")


if __name__ == "__main__":
    main()
