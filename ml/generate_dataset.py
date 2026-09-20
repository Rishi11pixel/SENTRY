import os
import numpy as np

# ============================================================
# SENTRY DATASET GENERATOR - V2
#
# Normal operation is anchored to REAL ESP32 observations.
#
# IMPORTANT:
# Move Gel / Harpic are NOT threat labels.
# Threat classes below are synthetic demonstration classes.
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = "ml/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = [
    "SAFE",
    "WEATHER",
    "ALCOHOL",
    "EXPLOSIVE",
    "NARCOTIC",
]


# ============================================================
# REAL HARDWARE NORMAL REGION
#
# Taken from the clean ESP32 stream you just provided.
#
# The hardware is currently around:
#
# MQ2      : 0.41 - 0.49
# MQ3      : 2.57 - 2.65
# MQ135    : 0.36 - 0.40
# SEN0567  : 1.50 - 1.86
# dV/dt    : 0.03 - 0.43
# Temp     : ~31.1
# Humidity : ~68
#
# We deliberately give SAFE a broad envelope.
# ============================================================

SAFE_CENTER = np.array([
    0.455,
    2.615,
    0.375,
    1.68,
    0.12,
    31.1,
    68.0
])


# ============================================================
# SAFE
#
# Broad real-hardware operating region.
# This is intentionally much wider than before.
# ============================================================

def generate_safe(n):

    samples = []

    for _ in range(n):

        vmq2 = rng.normal(
            0.455,
            0.030
        )

        vmq3 = rng.normal(
            2.615,
            0.035
        )

        vmq135 = rng.normal(
            0.375,
            0.020
        )

        # SEN0567 has shown substantial normal movement.
        sen0567 = rng.normal(
            1.68,
            0.12
        )

        d_vdt = abs(
            rng.normal(
                0.13,
                0.09
            )
        )

        temperature = rng.normal(
            31.1,
            0.45
        )

        humidity = rng.normal(
            68.0,
            2.0
        )

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            d_vdt,
            temperature,
            humidity
        ])

    return np.array(samples)


# ============================================================
# WEATHER
#
# Slow environmental drift around the real hardware region.
# ============================================================

def generate_weather(n):

    samples = []

    for _ in range(n):

        drift = rng.uniform(
            -1.0,
            1.0
        )

        vmq2 = (
            0.455
            - 0.035 * drift
            + rng.normal(0, 0.018)
        )

        vmq3 = (
            2.615
            - 0.045 * drift
            + rng.normal(0, 0.020)
        )

        vmq135 = (
            0.375
            + 0.018 * drift
            + rng.normal(0, 0.015)
        )

        sen0567 = (
            1.68
            - 0.14 * drift
            + rng.normal(0, 0.055)
        )

        d_vdt = abs(
            rng.normal(
                0.16,
                0.09
            )
        )

        temperature = (
            31.1
            + 2.5 * drift
            + rng.normal(0, 0.35)
        )

        humidity = (
            68.0
            - 7.0 * drift
            + rng.normal(0, 1.5)
        )

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            d_vdt,
            temperature,
            humidity
        ])

    return np.array(samples)


# ============================================================
# SYNTHETIC ALCOHOL
#
# Deliberately outside the normal hardware envelope.
# ============================================================

def generate_alcohol(n):

    samples = []

    for _ in range(n):

        vmq2 = rng.normal(
            0.62,
            0.025
        )

        vmq3 = rng.normal(
            2.95,
            0.035
        )

        vmq135 = rng.normal(
            0.55,
            0.025
        )

        sen0567 = rng.normal(
            2.05,
            0.035
        )

        d_vdt = abs(
            rng.normal(
                0.32,
                0.07
            )
        )

        temperature = rng.normal(
            31.1,
            0.5
        )

        humidity = rng.normal(
            68.0,
            2.0
        )

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            d_vdt,
            temperature,
            humidity
        ])

    return np.array(samples)


# ============================================================
# SYNTHETIC EXPLOSIVE
#
# Synthetic ML signature only.
# ============================================================

def generate_explosive(n):

    samples = []

    for _ in range(n):

        vmq2 = rng.normal(
            0.72,
            0.030
        )

        vmq3 = rng.normal(
            2.38,
            0.035
        )

        vmq135 = rng.normal(
            0.67,
            0.030
        )

        sen0567 = rng.normal(
            2.10,
            0.040
        )

        d_vdt = abs(
            rng.normal(
                0.40,
                0.08
            )
        )

        temperature = rng.normal(
            31.1,
            0.5
        )

        humidity = rng.normal(
            68.0,
            2.0
        )

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            d_vdt,
            temperature,
            humidity
        ])

    return np.array(samples)


# ============================================================
# SYNTHETIC NARCOTIC
#
# IMPORTANT:
# This is NOT based on actual narcotic measurements.
# It is only a synthetic demonstration class.
#
# It is deliberately kept away from the real normal region.
# ============================================================

def generate_narcotic(n):

    samples = []

    for _ in range(n):

        vmq2 = rng.normal(
            0.70,
            0.030
        )

        vmq3 = rng.normal(
            2.30,
            0.035
        )

        vmq135 = rng.normal(
            0.62,
            0.030
        )

        sen0567 = rng.normal(
            2.15,
            0.040
        )

        d_vdt = abs(
            rng.normal(
                0.38,
                0.08
            )
        )

        temperature = rng.normal(
            31.1,
            0.5
        )

        humidity = rng.normal(
            68.0,
            2.0
        )

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            d_vdt,
            temperature,
            humidity
        ])

    return np.array(samples)


# ============================================================
# DATASET SIZE
# ============================================================

N_PER_CLASS = 6000


# ============================================================
# GENERATE
# ============================================================

print("Generating SENTRY dataset V2...")

safe = generate_safe(N_PER_CLASS)
weather = generate_weather(N_PER_CLASS)
alcohol = generate_alcohol(N_PER_CLASS)
explosive = generate_explosive(N_PER_CLASS)
narcotic = generate_narcotic(N_PER_CLASS)


X = np.vstack([
    safe,
    weather,
    alcohol,
    explosive,
    narcotic
])


y = np.concatenate([
    np.full(N_PER_CLASS, 0),
    np.full(N_PER_CLASS, 1),
    np.full(N_PER_CLASS, 2),
    np.full(N_PER_CLASS, 3),
    np.full(N_PER_CLASS, 4)
])


# ============================================================
# CLIP VALUES
# ============================================================

X[:, 0] = np.clip(
    X[:, 0],
    0.20,
    1.00
)

X[:, 1] = np.clip(
    X[:, 1],
    2.10,
    3.20
)

X[:, 2] = np.clip(
    X[:, 2],
    0.15,
    0.90
)

X[:, 3] = np.clip(
    X[:, 3],
    1.20,
    2.40
)

X[:, 4] = np.clip(
    X[:, 4],
    0.0,
    0.80
)

X[:, 5] = np.clip(
    X[:, 5],
    20.0,
    40.0
)

X[:, 6] = np.clip(
    X[:, 6],
    20.0,
    95.0
)


# ============================================================
# SHUFFLE
# ============================================================

indices = rng.permutation(
    len(X)
)

X = X[indices]
y = y[indices]


# ============================================================
# SAVE
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X.npy"
    ),
    X
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y.npy"
    ),
    y
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "label_classes.npy"
    ),
    np.array(CLASSES)
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 60)
print("DATASET GENERATED")
print("=" * 60)

print(
    f"Samples  : {len(X)}"
)

print(
    f"Features : {X.shape[1]}"
)

print(
    f"Classes  : {CLASSES}"
)

print()

for i, name in enumerate(CLASSES):

    print(
        f"{name:10s}: "
        f"{np.sum(y == i)}"
    )

print()
print("Feature ranges:")

feature_names = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0567",
    "dVdt_max",
    "temperature",
    "humidity"
]

for i, name in enumerate(
    feature_names
):

    print(
        f"{name:12s}: "
        f"{X[:, i].min():.4f}"
        f" -> "
        f"{X[:, i].max():.4f}"
    )

print()
print(
    f"Saved to: {OUTPUT_DIR}/"
)