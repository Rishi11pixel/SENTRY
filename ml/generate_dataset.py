# ml/generate_dataset.py

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

N_PER_CLASS = 6000

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "dVdt_max",
    "temperature",
    "humidity",
]

LABELS = [
    "SAFE",
    "WEATHER",
    "ALCOHOL",
    "EXPLOSIVE",
    "NARCOTIC",
]


def clip_features(X):
    X[:, 0] = np.clip(X[:, 0], 0.20, 1.20)   # MQ2
    X[:, 1] = np.clip(X[:, 1], 1.50, 3.50)   # MQ3
    X[:, 2] = np.clip(X[:, 2], 0.15, 1.00)   # MQ135
    X[:, 3] = np.clip(X[:, 3], 0.00, 1.20)   # dVdt
    X[:, 4] = np.clip(X[:, 4], 20.0, 40.0)   # temperature
    X[:, 5] = np.clip(X[:, 5], 30.0, 90.0)   # humidity
    return X


# ============================================================
# SAFE
# Based around the actual three-MQ hardware operating region.
# ============================================================

safe_mean = np.array([
    0.455,   # VMQ2
    2.615,   # VMQ3
    0.375,   # VMQ135
    0.12,    # dVdt_max
    31.1,    # temperature
    68.0     # humidity
])

safe_std = np.array([
    0.035,
    0.045,
    0.025,
    0.09,
    0.50,
    2.0
])

safe = rng.normal(
    loc=safe_mean,
    scale=safe_std,
    size=(N_PER_CLASS, len(FEATURES))
)

safe = clip_features(safe)


# ============================================================
# WEATHER
# Environmental changes / gradual sensor drift.
# Not a chemical threat.
# ============================================================

drift = rng.uniform(0.0, 1.0, N_PER_CLASS)

weather = np.column_stack([
    0.455 - 0.035 * drift + rng.normal(0, 0.020, N_PER_CLASS),
    2.615 - 0.050 * drift + rng.normal(0, 0.025, N_PER_CLASS),
    0.375 + 0.020 * drift + rng.normal(0, 0.018, N_PER_CLASS),
    np.abs(rng.normal(0.16, 0.09, N_PER_CLASS)),
    31.1 + 2.5 * drift + rng.normal(0, 0.40, N_PER_CLASS),
    68.0 - 7.0 * drift + rng.normal(0, 1.5, N_PER_CLASS),
])

weather = clip_features(weather)


# ============================================================
# ALCOHOL
# Synthetic response pattern.
# ============================================================

alcohol_mean = np.array([
    0.62,   # MQ2
    2.95,   # MQ3
    0.55,   # MQ135
    0.32,   # dVdt
    31.1,
    68.0
])

alcohol_std = np.array([
    0.055,
    0.070,
    0.045,
    0.13,
    0.50,
    2.0
])

alcohol = rng.normal(
    loc=alcohol_mean,
    scale=alcohol_std,
    size=(N_PER_CLASS, len(FEATURES))
)

alcohol = clip_features(alcohol)


# ============================================================
# EXPLOSIVE
# Synthetic response pattern.
# ============================================================

explosive_mean = np.array([
    0.72,
    2.38,
    0.67,
    0.40,
    31.1,
    68.0
])

explosive_std = np.array([
    0.060,
    0.070,
    0.050,
    0.15,
    0.50,
    2.0
])

explosive = rng.normal(
    loc=explosive_mean,
    scale=explosive_std,
    size=(N_PER_CLASS, len(FEATURES))
)

explosive = clip_features(explosive)


# ============================================================
# NARCOTIC
# Synthetic response pattern.
# ============================================================

narcotic_mean = np.array([
    0.70,
    2.30,
    0.62,
    0.38,
    31.1,
    68.0
])

narcotic_std = np.array([
    0.060,
    0.070,
    0.050,
    0.14,
    0.50,
    2.0
])

narcotic = rng.normal(
    loc=narcotic_mean,
    scale=narcotic_std,
    size=(N_PER_CLASS, len(FEATURES))
)

narcotic = clip_features(narcotic)


# ============================================================
# BUILD DATASET
# ============================================================

datasets = []

for X, label in [
    (safe, "SAFE"),
    (weather, "WEATHER"),
    (alcohol, "ALCOHOL"),
    (explosive, "EXPLOSIVE"),
    (narcotic, "NARCOTIC"),
]:
    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = label
    datasets.append(df)


dataset = pd.concat(datasets, ignore_index=True)

# Shuffle
dataset = dataset.sample(
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output_path = Path(__file__).resolve().parent / "dataset.csv"

dataset.to_csv(output_path, index=False)


# ============================================================
# REPORT
# ============================================================

print("\n=== SENTRY 6-FEATURE DATASET ===")

print(f"Saved to: {output_path}")
print(f"Rows: {len(dataset)}")
print(f"Features: {len(FEATURES)}")

print("\nColumns:")
print(list(dataset.columns))

print("\nClass distribution:")
print(dataset["label"].value_counts())

print("\nFeature ranges:")
print(dataset[FEATURES].agg(["min", "max"]).T)

print("\nFirst 5 rows:")
print(dataset.head())