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
    X[:, 0] = np.clip(X[:, 0], 0.20, 1.20)
    X[:, 1] = np.clip(X[:, 1], 1.50, 3.50)
    X[:, 2] = np.clip(X[:, 2], 0.15, 1.00)
    X[:, 3] = np.clip(X[:, 3], 0.00, 1.20)
    X[:, 4] = np.clip(X[:, 4], 20.0, 40.0)
    X[:, 5] = np.clip(X[:, 5], 30.0, 90.0)
    return X


# ============================================================
# SAFE
# ============================================================

safe_mean = np.array([
    0.455,
    2.615,
    0.375,
    0.12,
    31.1,
    68.0
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
# ============================================================

alcohol_mean = np.array([
    0.62,
    2.95,
    0.55,
    0.32,
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
# Synthetic hazardous signature
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
# Synthetic hazardous signature
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
# CONTROLLED SENSOR OVERLAP
#
# IMPORTANT:
# These samples are STILL labelled SAFE/WEATHER.
#
# They represent benign/ambiguous sensor responses that
# partially overlap hazardous regions because gas sensors
# are cross-sensitive.
#
# We do NOT label these as household products.
# ============================================================

def make_overlap_samples(base_mean, target_mean, n, overlap):
    base_mean = np.asarray(base_mean)
    target_mean = np.asarray(target_mean)

    mean = (
        (1.0 - overlap) * base_mean
        + overlap * target_mean
    )

    std = np.array([
        0.045,   # VMQ2
        0.060,   # VMQ3
        0.040,   # VMQ135
        0.11,    # dVdt_max
        0.55,    # temperature
        2.2,     # humidity
    ])

    X = rng.normal(
        loc=mean,
        scale=std,
        size=(n, len(FEATURES))
    )

    return clip_features(X)


# ------------------------------------------------------------
# EXPLOSIVE-BOUNDARY OVERLAP
# ------------------------------------------------------------

EXPLOSIVE_OVERLAP = 1000

safe_explosive_overlap = make_overlap_samples(
    safe_mean,
    explosive_mean,
    EXPLOSIVE_OVERLAP,
    overlap=0.22
)

weather_explosive_overlap = make_overlap_samples(
    weather.mean(axis=0),
    explosive_mean,
    EXPLOSIVE_OVERLAP,
    overlap=0.20
)


# ------------------------------------------------------------
# NARCOTIC-BOUNDARY OVERLAP
# ------------------------------------------------------------

NARCOTIC_OVERLAP = 1000

safe_narcotic_overlap = make_overlap_samples(
    safe_mean,
    narcotic_mean,
    NARCOTIC_OVERLAP,
    overlap=0.22
)

weather_narcotic_overlap = make_overlap_samples(
    weather.mean(axis=0),
    narcotic_mean,
    NARCOTIC_OVERLAP,
    overlap=0.20
)


# ============================================================
# INSERT OVERLAP INTO SAFE / WEATHER
#
# Total class size remains 6000.
# ============================================================

safe_regular_count = (
    N_PER_CLASS
    - len(safe_explosive_overlap)
    - len(safe_narcotic_overlap)
)

weather_regular_count = (
    N_PER_CLASS
    - len(weather_explosive_overlap)
    - len(weather_narcotic_overlap)
)

safe = np.vstack([
    safe[:safe_regular_count],
    safe_explosive_overlap,
    safe_narcotic_overlap,
])

weather = np.vstack([
    weather[:weather_regular_count],
    weather_explosive_overlap,
    weather_narcotic_overlap,
])

safe = clip_features(safe)
weather = clip_features(weather)


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


dataset = pd.concat(
    datasets,
    ignore_index=True
)


# Shuffle
dataset = dataset.sample(
    frac=1.0,
    random_state=SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output_path = Path(__file__).resolve().parent / "dataset.csv"

dataset.to_csv(
    output_path,
    index=False
)


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

print("\nOverlap samples:")
print(f"  SAFE/EXPLOSIVE boundary : {EXPLOSIVE_OVERLAP}")
print(f"  WEATHER/EXPLOSIVE       : {EXPLOSIVE_OVERLAP}")
print(f"  SAFE/NARCOTIC boundary  : {NARCOTIC_OVERLAP}")
print(f"  WEATHER/NARCOTIC        : {NARCOTIC_OVERLAP}")

print("\nFirst 5 rows:")
print(dataset.head())