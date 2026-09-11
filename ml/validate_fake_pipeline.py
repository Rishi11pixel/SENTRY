from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "sentry_fake_readings_1800.txt"
TEST_PATH = ROOT / "data" / "raw" / "sentry_synthetic_test.csv"
MODEL_DIR = ROOT / "models" / "tinyml"
FEATURE_NAMES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0567",
    "dVdt_max",
    "temperature",
    "humidity",
]
READING_PATTERN = re.compile(
    r"Temperature:\s*([-+]?\d+(?:\.\d+)?)\s*C\s*"
    r"Humidity:\s*([-+]?\d+(?:\.\d+)?)\s*%\s*"
    r"MQ-2\s*\(Gas\):\s*(\d+)\s*"
    r"MQ-3\s*\(Flying-Fish\):\s*(\d+)\s*"
    r"MQ-135\s*\(Air Quality\):\s*(\d+)\s*"
    r"Fermion NH3\s*\(SEN0567\):\s*([-+]?\d+(?:\.\d+)?)\s*"
    r"(?:STATUS:\s*([^\r\n]+))?",
    re.MULTILINE,
)


def load_readings() -> list[dict[str, float | str | None]]:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(SOURCE_PATH)
    matches = READING_PATTERN.findall(SOURCE_PATH.read_text(encoding="utf-8"))
    readings = []
    for index, match in enumerate(matches):
        readings.append(
            {
                "timestamp": round(index * 0.1, 1),
                "temperature": float(match[0]),
                "humidity": float(match[1]),
                "mq2": float(match[2]),
                "mq3": float(match[3]),
                "mq135": float(match[4]),
                "sen0567": float(match[5]),
                "source_status": match[6].strip() if match[6] else None,
            }
        )
    if len(readings) != 1800:
        raise ValueError(f"Expected 1,800 readings, found {len(readings)}")
    return readings


def load_test_features() -> np.ndarray:
    with TEST_PATH.open(newline="", encoding="utf-8") as file:
        rows = csv.DictReader(file)
        return np.asarray(
            [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
            dtype=float,
        )


def summary(name: str, values: np.ndarray) -> None:
    print(f"\n{name} feature distribution ({len(values)} windows):")
    for index, feature in enumerate(FEATURE_NAMES):
        column = values[:, index]
        percentiles = np.percentile(column, [5, 50, 95])
        print(
            f"  {feature:10s} min={column.min():.5f} "
            f"p05={percentiles[0]:.5f} median={percentiles[1]:.5f} "
            f"p95={percentiles[2]:.5f} max={column.max():.5f} "
            f"mean={column.mean():.5f} std={column.std():.5f}"
        )


def main() -> None:
    sys.path.insert(0, str(ROOT / "ml_test"))
    import ml_server

    readings = load_readings()
    windows = [readings[index : index + 30] for index in range(0, len(readings), 30)]
    features = np.asarray([ml_server.extract_features(window) for window in windows])
    test_features = load_test_features()
    scaled = (features - ml_server.scaler_mean) / ml_server.scaler_scale

    print(f"Canonical source: {SOURCE_PATH}")
    print(f"Readings: {len(readings)}; complete windows: {len(windows)}")
    print(f"Raw fields per reading: temperature, humidity, mq2, mq3, mq135, sen0567")
    print(f"Feature order: {FEATURE_NAMES}")
    print(f"Model input shape: {scaled.shape}")
    print(f"Saved scaler mean: {MODEL_DIR / 'scaler_mean.npy'}")
    print(f"Saved scaler scale: {MODEL_DIR / 'scaler_scale.npy'}")

    summary("Fake", features)
    summary("Training/test reference", test_features)

    print("\nRepresentative windows:")
    for window_number in [1, 2, 30, 31, 45, 60]:
        index = window_number - 1
        prediction, confidence, probabilities = ml_server.predict(features[index])
        print(f"\n  Window {window_number}:")
        print(f"    raw summary: {windows[index][0]} ... {windows[index][-1]}")
        print(f"    features: {dict(zip(FEATURE_NAMES, np.round(features[index], 6)))}")
        print(f"    scaled: {np.round(scaled[index], 6).tolist()}")
        print(f"    prediction: {prediction}; confidence: {confidence:.6f}")
        print(f"    probabilities: {probabilities}")

    print("\nFake feature range containment in reference test ranges:")
    for index, name in enumerate(FEATURE_NAMES):
        inside = features[:, index].min() >= test_features[:, index].min() and features[:, index].max() <= test_features[:, index].max()
        print(f"  {name}: {'inside' if inside else 'outside'}")


if __name__ == "__main__":
    main()
