import os
import numpy as np

# ============================================================
# SENTRY Synthetic Dataset Generator
#
# IMPORTANT:
# - SAFE / WEATHER are anchored to REAL ESP32 observations.
# - Move Gel / Harpic are NOT assigned to threat classes.
# - ALCOHOL / EXPLOSIVE / NARCOTIC are synthetic demo classes.
# - These threat signatures are ML simulation patterns, NOT
#   chemical identification/calibration data.
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = "ml/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAMES = [
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


# ============================================================
# REAL HARDWARE BASELINE
#
# Based on the clean ESP32 readings we observed.
# ============================================================

CLEAN_CENTER = np.array([
    0.48,   # VMQ2
    2.67,   # VMQ3
    0.40,   # VMQ135
    1.72,   # VSEN0567
    0.08,   # dVdt_max
    30.5,   # temperature
    65.0,   # humidity
])


# ============================================================
# SAFE
#
# Small natural sensor variation around actual clean readings.
# ============================================================

def generate_safe(n):
    samples = []

    for _ in range(n):
        vmq2 = rng.normal(0.48, 0.018)
        vmq3 = rng.normal(2.67, 0.018)
        vmq135 = rng.normal(0.40, 0.018)
        sen0567 = rng.normal(1.72, 0.025)

        # Normal sensor/environment movement.
        d_vdt = abs(rng.normal(0.07, 0.035))

        temperature = rng.normal(30.5, 0.45)
        humidity = rng.normal(65.0, 1.8)

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            d_vdt,
            temperature,
            humidity,
        ])

    return np.array(samples)


# ============================================================
# WEATHER
#
# Represents environmental drift rather than a threat.
# Kept close to the real WEATHER readings we observed.
# ============================================================

def generate_weather(n):
    samples = []

    for _ in range(n):
        drift = rng.uniform(-1.0, 1.0)

        vmq2 = 0.48 - 0.035 * drift + rng.normal(0, 0.015)
        vmq3 = 2.67 - 0.025 * drift + rng.normal(0, 0.018)
        vmq135 = 0.40 + 0.018 * drift + rng.normal(0, 0.018)
        sen0567 = 1.72 - 0.025 * drift + rng.normal(0, 0.022)

        # Weather can create slow sensor movement.
        d_vdt = abs(rng.normal(0.11, 0.045))

        temperature = 30.5 + 2.0 * drift + rng.normal(0, 0.35)
        humidity = 65.0 - 7.0 * drift + rng.normal(0, 1.5)

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            d_vdt,
            temperature,
            humidity,
        ])

    return np.array(samples)


# ============================================================
# SYNTHETIC THREAT CLASSES
#
# These are deliberately separated from SAFE/WEATHER.
# They are only synthetic ML signatures for demonstration.
# They do NOT represent measured explosive/narcotic chemistry.
# ============================================================

def generate_alcohol(n):
    samples = []

    for _ in range(n):
        # Stronger MQ3 response + supporting changes.
        vmq2 = rng.normal(0.57, 0.025)
        vmq3 = rng.normal(2.88, 0.035)
        vmq135 = rng.normal(0.51, 0.025)
        sen0567 = rng.normal(1.78, 0.035)

        d_vdt = rng.normal(0.24, 0.06)

        temperature = rng.normal(30.5, 0.5)
        humidity = rng.normal(65.0, 2.0)

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            abs(d_vdt),
            temperature,
            humidity,
        ])

    return np.array(samples)


def generate_explosive(n):
    samples = []

    for _ in range(n):
        # Synthetic multi-sensor response pattern.
        vmq2 = rng.normal(0.68, 0.035)
        vmq3 = rng.normal(2.58, 0.035)
        vmq135 = rng.normal(0.62, 0.035)
        sen0567 = rng.normal(1.84, 0.040)

        d_vdt = rng.normal(0.32, 0.07)

        temperature = rng.normal(30.5, 0.5)
        humidity = rng.normal(65.0, 2.0)

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            abs(d_vdt),
            temperature,
            humidity,
        ])

    return np.array(samples)


def generate_narcotic(n):
    samples = []

    for _ in range(n):
        # Another synthetic multi-sensor pattern,
        # intentionally different from SAFE/WEATHER.
        vmq2 = rng.normal(0.61, 0.030)
        vmq3 = rng.normal(2.43, 0.035)
        vmq135 = rng.normal(0.57, 0.030)
        sen0567 = rng.normal(1.91, 0.040)

        d_vdt = rng.normal(0.29, 0.065)

        temperature = rng.normal(30.5, 0.5)
        humidity = rng.normal(65.0, 2.0)

        samples.append([
            vmq2,
            vmq3,
            vmq135,
            sen0567,
            abs(d_vdt),
            temperature,
            humidity,
        ])

    return np.array(samples)


# ============================================================
# DATASET SIZE
# ============================================================

N_PER_CLASS = 6000


# ============================================================
# GENERATE
# ============================================================

print("Generating SENTRY dataset...")

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
    narcotic,
])

y = np.concatenate([
    np.full(N_PER_CLASS, 0),
    np.full(N_PER_CLASS, 1),
    np.full(N_PER_CLASS, 2),
    np.full(N_PER_CLASS, 3),
    np.full(N_PER_CLASS, 4),
])


# ============================================================
# SAFETY CLIPPING
# ============================================================

# Prevent synthetic noise from creating impossible values.
X[:, 0] = np.clip(X[:, 0], 0.20, 1.00)   # MQ2
X[:, 1] = np.clip(X[:, 1], 2.20, 3.20)   # MQ3
X[:, 2] = np.clip(X[:, 2], 0.15, 0.90)   # MQ135
X[:, 3] = np.clip(X[:, 3], 1.20, 2.30)   # SEN0567
X[:, 4] = np.clip(X[:, 4], 0.0, 0.80)    # dV/dt
X[:, 5] = np.clip(X[:, 5], 20.0, 40.0)   # temperature
X[:, 6] = np.clip(X[:, 6], 20.0, 95.0)   # humidity


# ============================================================
# SHUFFLE
# ============================================================

indices = rng.permutation(len(X))

X = X[indices]
y = y[indices]


# ============================================================
# SAVE
# ============================================================

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
np.save(
    os.path.join(OUTPUT_DIR, "label_classes.npy"),
    np.array(CLASSES)
)


# ============================================================
# REPORT
# ============================================================

print()
print("Dataset generated successfully.")
print(f"Samples      : {len(X)}")
print(f"Features     : {X.shape[1]}")
print(f"Classes      : {CLASSES}")
print()

for i, name in enumerate(CLASSES):
    count = np.sum(y == i)
    print(f"{name:10s}: {count}")

print()
print("Feature ranges:")

for i, name in enumerate(FEATURE_NAMES):
    print(
        f"{name:12s}: "
        f"{X[:, i].min():.4f} -> {X[:, i].max():.4f}"
    )

print()
print(f"Saved to: {OUTPUT_DIR}/X.npy")
print(f"Saved to: {OUTPUT_DIR}/y.npy")
print(f"Saved to: {OUTPUT_DIR}/label_classes.npy")