from __future__ import annotations

import json
from pathlib import Path

import joblib
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
RF_MODEL_PATH = ROOT / "models" / "random_forest" / "random_forest.joblib"
RF_ENCODER_PATH = ROOT / "models" / "random_forest" / "label_encoder.joblib"
TINYML_MODEL_PATH = ROOT / "models" / "tinyml" / "sentry_tinyml.keras"
TINYML_MEAN_PATH = ROOT / "models" / "tinyml" / "scaler_mean.npy"
TINYML_SCALE_PATH = ROOT / "models" / "tinyml" / "scaler_scale.npy"
TINYML_LABELS_PATH = ROOT / "models" / "tinyml" / "label_classes.npy"
RESULT_DIR = ROOT / "results"
JSON_PATH = RESULT_DIR / "final_evaluation.json"
PLOT_PATH = RESULT_DIR / "final_confusion_matrices.png"

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0568",
    "dVdt_max",
    "temperature",
    "humidity",
]

CLASSES = [
    "ALCOHOL_SANITIZER",
    "AMBIENT_CLEAN",
    "EXPLOSIVE_PROXY",
    "NARCOTIC_PROXY",
    "WEATHER_DRIFT",
]


def load_final_test_data() -> pd.DataFrame:
    df = pd.read_csv(TEST_DATA_PATH)

    if len(df) != 12000:
        raise ValueError(f"Expected exactly 12,000 rows in final test set, found {len(df)}.")

    expected_distribution = pd.Series({label: 2400 for label in CLASSES})
    actual_distribution = df["label"].value_counts().reindex(CLASSES, fill_value=0)

    if not actual_distribution.equals(expected_distribution):
        raise ValueError(
            "Final test set class distribution is not exactly 2,400 samples per class. "
            f"Actual distribution: {actual_distribution.to_dict()}"
        )

    return df


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=np.arange(len(labels)))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.astype(int).tolist(),
    }
    return metrics


def evaluate_random_forest(X_test: pd.DataFrame, y_true_labels: np.ndarray) -> tuple[dict, np.ndarray]:
    model = joblib.load(RF_MODEL_PATH)
    encoder = joblib.load(RF_ENCODER_PATH)

    predictions = model.predict(X_test)
    prediction_labels = encoder.inverse_transform(predictions)
    true_encoded = encoder.transform(y_true_labels)
    predicted_encoded = encoder.transform(prediction_labels)

    metrics = compute_classification_metrics(
        y_true=true_encoded,
        y_pred=predicted_encoded,
        labels=CLASSES,
    )

    return metrics, confusion_matrix(true_encoded, predicted_encoded, labels=np.arange(len(CLASSES)))


def evaluate_tinyml(X_test: np.ndarray, y_true: np.ndarray) -> tuple[dict, np.ndarray]:
    scaler_mean = np.load(TINYML_MEAN_PATH)
    scaler_scale = np.load(TINYML_SCALE_PATH)
    saved_classes = np.load(TINYML_LABELS_PATH, allow_pickle=True)

    saved_classes_list = list(saved_classes)
    if saved_classes_list != CLASSES:
        raise ValueError(
            "Saved TinyML label order does not match the required class order. "
            f"Expected {CLASSES}, got {saved_classes_list}"
        )

    X_test_scaled = (X_test - scaler_mean) / scaler_scale
    X_test_scaled = X_test_scaled.astype(np.float32)

    model = tf.keras.models.load_model(TINYML_MODEL_PATH)
    probabilities = model.predict(X_test_scaled, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    y_true_encoded = np.asarray(y_true, dtype=int)
    metrics = compute_classification_metrics(
        y_true=y_true_encoded,
        y_pred=predictions,
        labels=CLASSES,
    )

    return metrics, confusion_matrix(y_true_encoded, predictions, labels=np.arange(len(CLASSES)))


def plot_confusion_matrices(rf_cm: np.ndarray, tinyml_cm: np.ndarray) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def plot_single(ax, cm: np.ndarray, title: str) -> None:
        cmap = plt.get_cmap("Blues")
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_title(title)
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

    plot_single(axes[0], rf_cm, "Random Forest — Final Test")
    plot_single(axes[1], tinyml_cm, "TinyML Float32 — Final Test")

    fig.suptitle("SENTRY Final Blind Evaluation — Confusion Matrices", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_results(label: str, metrics: dict) -> None:
    print(f"\n{label}:")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall: {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%")
    print("Per-class metrics:")
    for class_name in CLASSES:
        class_metrics = metrics["per_class"][class_name]
        print(
            f"  {class_name}: precision={class_metrics['precision']:.4f}, "
            f"recall={class_metrics['recall']:.4f}, f1={class_metrics['f1']:.4f}, "
            f"support={class_metrics['support']}"
        )
    print("Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print("  " + " ".join(f"{value:4d}" for value in row))


def main() -> None:
    df = load_final_test_data()
    X = df[FEATURES]
    y_labels = df["label"].to_numpy()

    rf_metrics, rf_cm = evaluate_random_forest(X, y_labels)
    y_encoded = np.array([CLASSES.index(label) for label in y_labels], dtype=np.int64)
    tinyml_metrics, tinyml_cm = evaluate_tinyml(X.to_numpy(dtype=np.float32), y_encoded)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": {
            "path": str(TEST_DATA_PATH.relative_to(ROOT)),
            "rows": int(len(df)),
            "class_distribution": {label: int(df["label"].value_counts().get(label, 0)) for label in CLASSES},
            "feature_order": FEATURES,
            "class_order": CLASSES,
            "final_test_set_used_only_for_evaluation": True,
        },
        "random_forest": rf_metrics,
        "tinyml_float32": tinyml_metrics,
        "comparison": {
            "random_forest_accuracy_percent": float(rf_metrics["accuracy"] * 100.0),
            "random_forest_macro_f1_percent": float(rf_metrics["macro_f1"] * 100.0),
            "tinyml_accuracy_percent": float(tinyml_metrics["accuracy"] * 100.0),
            "tinyml_macro_f1_percent": float(tinyml_metrics["macro_f1"] * 100.0),
        },
    }

    with JSON_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)

    plot_confusion_matrices(rf_cm, tinyml_cm)

    print("SENTRY Final Blind Evaluation")
    print(f"Dataset: {TEST_DATA_PATH.relative_to(ROOT)}")
    print(f"Rows: {len(df)}")
    print(f"Class distribution: { {label: int(df['label'].value_counts().get(label, 0)) for label in CLASSES} }")
    print("Final test set was used ONLY for evaluation.")

    print_results("Random Forest", rf_metrics)
    print_results("TinyML Float32", tinyml_metrics)

    print("\nFinal comparison:")
    print(f"Random Forest:\nAccuracy: {rf_metrics['accuracy'] * 100:.2f}%\nMacro F1: {rf_metrics['macro_f1'] * 100:.2f}%")
    print(f"\nTinyML Float32:\nAccuracy: {tinyml_metrics['accuracy'] * 100:.2f}%\nMacro F1: {tinyml_metrics['macro_f1'] * 100:.2f}%")

    print(f"\nSaved evaluation report to: {JSON_PATH}")
    print(f"Saved confusion matrices to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
