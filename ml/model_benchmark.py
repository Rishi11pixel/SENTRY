from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

try:
	from xgboost import XGBClassifier
except ImportError as exc:
	raise ImportError(
		"XGBoost is required for ml/model_benchmark.py. Install it with: "
		"python -m pip install xgboost"
	) from exc


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "raw" / "sentry_synthetic_dataset.csv"
TEST_PATH = ROOT / "data" / "raw" / "sentry_synthetic_test.csv"
RESULTS_DIR = ROOT / "results"
JSON_PATH = RESULTS_DIR / "model_benchmark_results.json"
CSV_PATH = RESULTS_DIR / "model_benchmark_results.csv"
PLOT_PATH = RESULTS_DIR / "model_benchmark_confusion_matrices.png"

FEATURES = [
	"VMQ2",
	"VMQ3",
	"VMQ135",
	"VSEN0567",
	"dVdt_max",
	"temperature",
	"humidity",
]

CLASSES = ["SAFE", "WEATHER", "ALCOHOL", "EXPLOSIVE", "NARCOTIC"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
THREAT_CLASSES = {"EXPLOSIVE", "NARCOTIC"}


def load_frozen_data() -> tuple[pd.DataFrame, pd.DataFrame]:
	train_df = pd.read_csv(TRAIN_PATH)
	test_df = pd.read_csv(TEST_PATH)
	expected_columns = FEATURES + ["label"]

	for name, frame, expected_rows in (
		("training", train_df, 60_000),
		("final test", test_df, 12_000),
	):
		if len(frame) != expected_rows:
			raise ValueError(f"Expected {expected_rows:,} {name} rows, found {len(frame):,}.")
		if frame.columns.tolist() != expected_columns:
			raise ValueError(f"Unexpected {name} columns: {frame.columns.tolist()}")
		if set(frame["label"]) != set(CLASSES):
			raise ValueError(f"Unexpected {name} labels: {sorted(frame['label'].unique())}")

	expected_train_counts = {label: 12_000 for label in CLASSES}
	expected_test_counts = {label: 2_400 for label in CLASSES}
	if train_df["label"].value_counts().to_dict() != expected_train_counts:
		raise ValueError("Training class distribution is not exactly 12,000 per class.")
	if test_df["label"].value_counts().to_dict() != expected_test_counts:
		raise ValueError("Final test class distribution is not exactly 2,400 per class.")

	return train_df, test_df


def build_models() -> dict[str, object]:
	return {
		"Random Forest": RandomForestClassifier(
			n_estimators=150,
			max_depth=8,
			random_state=42,
			n_jobs=-1,
		),
		"XGBoost": XGBClassifier(
			n_estimators=150,
			max_depth=6,
			learning_rate=0.08,
			subsample=0.9,
			colsample_bytree=0.9,
			objective="multi:softprob",
			num_class=5,
			eval_metric="mlogloss",
			random_state=42,
			n_jobs=-1,
			tree_method="hist",
		),
		"HistGradientBoosting": HistGradientBoostingClassifier(
			max_iter=150,
			learning_rate=0.08,
			max_leaf_nodes=31,
			random_state=42,
		),
		"SVM RBF": SVC(kernel="rbf", C=2.0, gamma="scale"),
		"Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
		"KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
		"Extra Trees": ExtraTreesClassifier(
			n_estimators=150,
			max_depth=8,
			random_state=42,
			n_jobs=-1,
		),
	}


def native_value(value):
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, (np.integer, int)):
		return int(value)
	if isinstance(value, (np.floating, float)):
		return float(value)
	return value


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
	precision, recall, f1, support = precision_recall_fscore_support(
		y_true,
		y_pred,
		labels=np.arange(len(CLASSES)),
		zero_division=0,
	)
	true_threat = np.isin(y_true, [CLASS_TO_INDEX[label] for label in THREAT_CLASSES])
	predicted_threat = np.isin(y_pred, [CLASS_TO_INDEX[label] for label in THREAT_CLASSES])
	binary_cm = confusion_matrix(true_threat, predicted_threat, labels=[False, True])
	tn, fp, fn, tp = binary_cm.ravel()
	threat_recall = tp / (tp + fn) if tp + fn else 0.0
	threat_fnr = fn / (tp + fn) if tp + fn else 0.0
	threat_precision = tp / (tp + fp) if tp + fp else 0.0
	threat_f1 = 2 * threat_precision * threat_recall / (threat_precision + threat_recall) if threat_precision + threat_recall else 0.0

	return {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
		"per_class": {
			label: {
				"precision": float(precision[index]),
				"recall": float(recall[index]),
				"f1": float(f1[index]),
				"support": int(support[index]),
			}
			for index, label in enumerate(CLASSES)
		},
		"confusion_matrix": confusion_matrix(
			y_true, y_pred, labels=np.arange(len(CLASSES))
		).astype(int).tolist(),
		"threat_safety": {
			"threat_definition": "EXPLOSIVE + NARCOTIC",
			"non_threat_definition": "SAFE + WEATHER + ALCOHOL",
			"TP": int(tp),
			"TN": int(tn),
			"FP": int(fp),
			"FN": int(fn),
			"threat_recall": float(threat_recall),
			"threat_false_negative_rate": float(threat_fnr),
			"threat_precision": float(threat_precision),
			"threat_f1": float(threat_f1),
			"explosive_recall": float(recall[CLASS_TO_INDEX["EXPLOSIVE"]]),
			"narcotic_recall": float(recall[CLASS_TO_INDEX["NARCOTIC"]]),
		},
	}


def plot_confusion_matrices(results: dict[str, dict]) -> None:
	fig, axes = plt.subplots(2, 4, figsize=(18, 9))
	axes = axes.ravel()
	cmap = plt.get_cmap("viridis")

	for axis, (name, result) in zip(axes, results.items()):
		matrix = np.asarray(result["confusion_matrix"])
		image = axis.imshow(matrix, cmap=cmap, interpolation="nearest")
		axis.set_title(name)
		axis.set_xlabel("Predicted label")
		axis.set_ylabel("True label")
		axis.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
		axis.set_yticks(range(len(CLASSES)), CLASSES)
		for row in range(len(CLASSES)):
			for column in range(len(CLASSES)):
				value = matrix[row, column]
				red, green, blue, _ = cmap(value / max(1, matrix.max()))
				luminance = 0.299 * red + 0.587 * green + 0.114 * blue
				axis.text(
					column,
					row,
					f"{value:,}",
					ha="center",
					va="center",
					color="white" if luminance < 0.55 else "black",
					fontsize=8,
					fontweight="bold",
				)
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

	axes[-1].axis("off")
	fig.suptitle("SENTRY Conventional ML Benchmark — Final Test Confusion Matrices")
	fig.tight_layout()
	fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	train_df, test_df = load_frozen_data()
	X = train_df[FEATURES].to_numpy(dtype=np.float32)
	y = np.asarray([CLASS_TO_INDEX[label] for label in train_df["label"]], dtype=np.int64)
	X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
	y_test = np.asarray([CLASS_TO_INDEX[label] for label in test_df["label"]], dtype=np.int64)

	X_train, X_validation, y_train, y_validation = train_test_split(
		X,
		y,
		test_size=0.20,
		random_state=42,
		stratify=y,
	)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_validation = scaler.transform(X_validation)
	X_test = scaler.transform(X_test)

	models = build_models()
	validation_results = {}
	test_results = {}

	print("SENTRY conventional ML benchmark")
	print(f"Training samples: {len(X_train):,}")
	print(f"Validation samples: {len(X_validation):,}")
	print(f"Final test samples: {len(X_test):,}")
	print("Final test set is used only for final evaluation.")

	for name, model in models.items():
		print(f"\nTraining {name}...")
		model.fit(X_train, y_train)
		validation_predictions = model.predict(X_validation)
		test_predictions = model.predict(X_test)
		validation_results[name] = calculate_metrics(y_validation, validation_predictions)
		test_results[name] = calculate_metrics(y_test, test_predictions)

	sorted_models = sorted(
		test_results,
		key=lambda name: test_results[name]["threat_safety"]["threat_false_negative_rate"],
	)
	comparison_rows = []
	for rank, name in enumerate(sorted_models, start=1):
		metrics = test_results[name]
		safety = metrics["threat_safety"]
		comparison_rows.append(
			{
				"rank_by_threat_fnr": rank,
				"model": name,
				"accuracy": metrics["accuracy"],
				"macro_f1": metrics["macro_f1"],
				"threat_fnr": safety["threat_false_negative_rate"],
				"threat_recall": safety["threat_recall"],
				"threat_precision": safety["threat_precision"],
				"threat_f1": safety["threat_f1"],
				"explosive_recall": safety["explosive_recall"],
				"narcotic_recall": safety["narcotic_recall"],
				"TP": safety["TP"],
				"TN": safety["TN"],
				"FP": safety["FP"],
				"FN": safety["FN"],
			}
		)

	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(comparison_rows).to_csv(CSV_PATH, index=False)
	plot_confusion_matrices(test_results)

	report = {
		"dataset": {
			"training_path": str(TRAIN_PATH.relative_to(ROOT)),
			"final_test_path": str(TEST_PATH.relative_to(ROOT)),
			"training_rows": int(len(train_df)),
			"final_test_rows": int(len(test_df)),
			"features": FEATURES,
			"classes": CLASSES,
			"split": "80/20 stratified train/validation, random_state=42",
			"final_test_used_only_for_evaluation": True,
		},
		"validation": validation_results,
		"final_test": test_results,
		"comparison_sorted_by_threat_fnr": comparison_rows,
	}
	with JSON_PATH.open("w", encoding="utf-8") as json_file:
		json.dump(report, json_file, indent=2)

	print("\nFinal comparison sorted by Threat FNR (lower is better):")
	print(pd.DataFrame(comparison_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
	print("\nSeparate class recall:")
	for name in sorted_models:
		safety = test_results[name]["threat_safety"]
		print(f"{name}: EXPLOSIVE={safety['explosive_recall']:.4f}, NARCOTIC={safety['narcotic_recall']:.4f}")
	print(f"\nSaved JSON: {JSON_PATH}")
	print(f"Saved CSV: {CSV_PATH}")
	print(f"Saved confusion matrices: {PLOT_PATH}")


if __name__ == "__main__":
	main()
