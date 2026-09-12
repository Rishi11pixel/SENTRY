from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "model_benchmark_results.json"
OUTPUT_DIR = ROOT / "results" / "plots"

EXPECTED_MODELS = [
    "Random Forest",
    "XGBoost",
    "HistGradientBoosting",
    "SVM RBF",
    "Logistic Regression",
    "KNN",
    "Extra Trees",
]

REQUIRED_FINAL_METRICS = ["accuracy", "macro_f1", "threat_safety"]
REQUIRED_SAFETY_METRICS = [
    "threat_recall",
    "threat_false_negative_rate",
    "TP",
    "TN",
    "FP",
    "FN",
]


def load_and_validate_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Benchmark results not found: {RESULTS_PATH}")

    with RESULTS_PATH.open(encoding="utf-8") as json_file:
        report = json.load(json_file)

    if "final_test" not in report:
        raise ValueError("Benchmark JSON is missing the final_test section.")

    final_test = report["final_test"]
    missing_models = [model for model in EXPECTED_MODELS if model not in final_test]
    if missing_models:
        raise ValueError(f"Benchmark JSON is missing models: {missing_models}")

    for model in EXPECTED_MODELS:
        model_metrics = final_test[model]
        missing_metrics = [
            metric for metric in REQUIRED_FINAL_METRICS if metric not in model_metrics
        ]
        if missing_metrics:
            raise ValueError(f"{model} is missing final-test metrics: {missing_metrics}")

        safety_metrics = model_metrics["threat_safety"]
        missing_safety = [
            metric for metric in REQUIRED_SAFETY_METRICS if metric not in safety_metrics
        ]
        if missing_safety:
            raise ValueError(f"{model} is missing threat-safety metrics: {missing_safety}")

    return final_test


def build_frame(final_test: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model in EXPECTED_MODELS:
        metrics = final_test[model]
        safety = metrics["threat_safety"]
        threat_total = int(safety["TP"]) + int(safety["FN"])
        calculated_recall = int(safety["TP"]) / threat_total if threat_total else 0.0
        calculated_fnr = int(safety["FN"]) / threat_total if threat_total else 0.0
        json_recall = float(safety["threat_recall"])
        json_fnr = float(safety["threat_false_negative_rate"])

        recall_difference = abs(json_recall - calculated_recall)
        fnr_difference = abs(json_fnr - calculated_fnr)
        if recall_difference > 1e-9:
            print(
                f"WARNING: {model} Threat Recall mismatch: "
                f"JSON={json_recall}, calculated={calculated_recall}, "
                f"difference={recall_difference}"
            )
        if fnr_difference > 1e-9:
            print(
                f"WARNING: {model} Threat FNR mismatch: "
                f"JSON={json_fnr}, calculated={calculated_fnr}, "
                f"difference={fnr_difference}"
            )
        if threat_total != 4800:
            raise ValueError(
                f"{model} has TP + FN = {threat_total}; expected 4,800 threat samples."
            )

        rows.append(
            {
                "Model": model,
                "Accuracy": float(metrics["accuracy"]),
                "Macro F1": float(metrics["macro_f1"]),
                "Threat Recall": json_recall,
                "Threat FNR": json_fnr,
                "TP": int(safety["TP"]),
                "TN": int(safety["TN"]),
                "FP": int(safety["FP"]),
                "FN": int(safety["FN"]),
            }
        )

    return pd.DataFrame(rows)


def annotate_horizontal_bars(axis, bars, values, number_format):
    maximum = max(values) if values else 0
    offset = maximum * 0.015 if maximum else 0.15
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            format(value, number_format),
            va="center",
            fontsize=9,
        )


def plot_threat_fnr(frame: pd.DataFrame) -> Path:
    ordered = frame.sort_values("Threat FNR", ascending=True)
    values = ordered["Threat FNR"] * 100.0
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(ordered["Model"], values)
    axis.set_title("Threat False Negative Rate by Model")
    axis.set_xlabel("Threat False Negative Rate (%)")
    axis.set_ylabel("Model")
    annotate_horizontal_bars(axis, bars, values.tolist(), ".2f")
    axis.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    path = OUTPUT_DIR / "threat_false_negative_rate.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threat_recall(frame: pd.DataFrame) -> Path:
    ordered = frame.sort_values("Threat Recall", ascending=False)
    values = ordered["Threat Recall"] * 100.0
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(ordered["Model"], values)
    axis.set_title("Threat Recall by Model")
    axis.set_xlabel("Threat Recall (%)")
    axis.set_ylabel("Model")
    axis.invert_yaxis()
    annotate_horizontal_bars(axis, bars, values.tolist(), ".2f")
    axis.set_xlim(0, 100)
    fig.tight_layout()
    path = OUTPUT_DIR / "threat_recall.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_accuracy_vs_macro_f1(frame: pd.DataFrame) -> Path:
    positions = np.arange(len(frame))
    width = 0.36
    accuracy = frame["Accuracy"] * 100.0
    macro_f1 = frame["Macro F1"] * 100.0
    fig, axis = plt.subplots(figsize=(12, 6))
    accuracy_bars = axis.bar(positions - width / 2, accuracy, width, label="Accuracy")
    f1_bars = axis.bar(positions + width / 2, macro_f1, width, label="Macro F1")
    axis.set_title("Overall Model Performance")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score (%)")
    axis.set_xticks(positions, frame["Model"], rotation=28, ha="right")
    axis.legend()
    for bars in (accuracy_bars, f1_bars):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{bar.get_height():.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_ylim(0, max(accuracy.max(), macro_f1.max()) * 1.12)
    fig.tight_layout()
    path = OUTPUT_DIR / "accuracy_vs_macro_f1.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_missed_threats(frame: pd.DataFrame) -> Path:
    ordered = frame.sort_values("FN", ascending=True)
    values = ordered["FN"].tolist()
    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(ordered["Model"], values)
    axis.set_title("Missed Threats by Model")
    axis.set_xlabel("False Negatives")
    axis.set_ylabel("Model")
    annotate_horizontal_bars(axis, bars, values, "d")
    axis.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    path = OUTPUT_DIR / "missed_threats.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threat_recall_vs_fnr(frame: pd.DataFrame) -> Path:
    positions = np.arange(len(frame))
    width = 0.36
    recall = frame["Threat Recall"] * 100.0
    fnr = frame["Threat FNR"] * 100.0
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(positions - width / 2, recall, width, label="Threat Recall")
    axis.bar(positions + width / 2, fnr, width, label="Threat FNR")
    axis.set_title("Threat Detection Performance")
    axis.set_xlabel("Model")
    axis.set_ylabel("Rate (%)")
    axis.set_xticks(positions, frame["Model"], rotation=28, ha="right")
    axis.set_ylim(0, 100)
    axis.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "threat_recall_vs_fnr.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    final_test = load_and_validate_results()
    frame = build_frame(final_test)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Benchmark metric verification:")
    print(f"Validated {len(frame)} models against final_test.")
    print("Verified TP + FN == 4800 for every model.")
    print("Threat Recall and Threat FNR were checked against TP/FN calculations.")

    generated_paths = [
        plot_threat_fnr(frame),
        plot_threat_recall(frame),
        plot_accuracy_vs_macro_f1(frame),
        plot_missed_threats(frame),
        plot_threat_recall_vs_fnr(frame),
    ]

    ranked = frame.sort_values("Threat FNR", ascending=True).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    display_frame = ranked[[
        "Rank",
        "Model",
        "Accuracy",
        "Macro F1",
        "Threat Recall",
        "Threat FNR",
        "FN",
    ]].copy()
    for column in ["Accuracy", "Macro F1", "Threat Recall", "Threat FNR"]:
        display_frame[column] = display_frame[column].map(lambda value: f"{value * 100:.2f}%")

    print("\nRank | Model | Accuracy | Macro F1 | Threat Recall | Threat FNR | FN")
    print(display_frame.to_string(index=False))
    print("\nVisualization complete.")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
