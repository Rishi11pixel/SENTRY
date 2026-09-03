from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEED = 87

TRAIN_PER_CLASS = 12_000
TEST_PER_CLASS = 2_400

N_SAMPLES = 30
DT = 0.1  # 100 ms

CLASSES = [
    "AMBIENT_CLEAN",
    "WEATHER_DRIFT",
    "ALCOHOL_SANITIZER",
    "EXPLOSIVE_PROXY",
    "NARCOTIC_PROXY",
]

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0568",
    "dVdt_max",
    "temperature",
    "humidity",
]

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw"

TRAIN_FILE = OUTPUT_DIR / "sentry_synthetic_dataset.csv"
TEST_FILE = OUTPUT_DIR / "sentry_synthetic_test.csv"

rng = np.random.default_rng(SEED)


# ============================================================
# SIMULATED SENSOR BASELINES
# ============================================================

# Approximate simulation values only.
# These are NOT claimed specifications for the physical sensors.

BASELINE = np.array([
    0.82,  # MQ-2
    0.74,  # MQ-3
    0.91,  # MQ-135
    0.68,  # SEN0568
])

# Small measurement noise.
NOISE = np.array([
    0.008,
    0.009,
    0.010,
    0.009,
])


# ============================================================
# HELPERS
# ============================================================

def correlated_noise(n, sigma, alpha=0.90):
    raw = rng.normal(0.0, sigma, n)

    output = np.zeros(n)
    output[0] = raw[0]

    for i in range(1, n):
        output[i] = (
            alpha * output[i - 1]
            + np.sqrt(1.0 - alpha ** 2) * raw[i]
        )

    return output


def sigmoid(t, center, width):
    return 1.0 / (
        1.0 + np.exp(-(t - center) / width)
    )


def pulse(t, center, width):
    return np.exp(
        -0.5 * ((t - center) / width) ** 2
    )


def clip_voltage(x):
    return np.clip(x, 0.02, 3.25)


def generate_environment():

    temperature = np.clip(
        rng.normal(24.0, 5.0),
        10.0,
        40.0,
    )

    humidity = np.clip(
        rng.normal(55.0, 15.0),
        20.0,
        90.0,
    )

    return temperature, humidity


# ============================================================
# GENERATE ONE SENSOR WINDOW
# ============================================================

def generate_window(label):

    # 30 samples at 100 ms = 3 seconds.
    t = np.arange(N_SAMPLES) * DT

    temperature, humidity = generate_environment()

    # --------------------------------------------------------
    # Sensor-to-sensor manufacturing variation
    # --------------------------------------------------------

    offsets = rng.normal(
        0.0,
        0.035,
        4,
    )

    gains = rng.normal(
        1.0,
        0.035,
        4,
    )

    sensors = np.tile(
        BASELINE,
        (N_SAMPLES, 1),
    )

    for sensor in range(4):

        sensors[:, sensor] += offsets[sensor]

        sensors[:, sensor] *= gains[sensor]

        sensors[:, sensor] += correlated_noise(
            N_SAMPLES,
            NOISE[sensor],
            alpha=0.92,
        )

    # --------------------------------------------------------
    # Environmental effects
    # --------------------------------------------------------

    humidity_factor = (
        humidity - 55.0
    ) / 35.0

    temperature_factor = (
        temperature - 24.0
    ) / 16.0

    humidity_coeff = np.array([
        0.018,
        0.014,
        0.022,
        0.016,
    ])

    temperature_coeff = np.array([
        0.008,
        0.007,
        0.009,
        0.008,
    ])

    sensors += (
        humidity_factor
        * humidity_coeff
    )

    sensors += (
        temperature_factor
        * temperature_coeff
    )

    # --------------------------------------------------------
    # Slow baseline drift
    # --------------------------------------------------------

    drift_amount = rng.uniform(
        -0.025,
        0.025,
    )

    sensors += (
        drift_amount
        * (t / t[-1])
    )[:, None]

    # Variable event start.
    onset = rng.uniform(
        0.45,
        1.70,
    )

    # ========================================================
    # AMBIENT CLEAN
    # ========================================================

    if label == "AMBIENT_CLEAN":

        sensors += correlated_noise(
            N_SAMPLES,
            0.003,
            alpha=0.97,
        )[:, None]

        # Rare harmless transient.
        if rng.random() < 0.12:

            center = rng.uniform(
                0.5,
                2.5,
            )

            width = rng.uniform(
                0.08,
                0.22,
            )

            amplitude = rng.uniform(
                0.008,
                0.035,
            )

            sensor = rng.integers(
                0,
                4,
            )

            sensors[:, sensor] += (
                amplitude
                * pulse(
                    t,
                    center,
                    width,
                )
            )

    # ========================================================
    # WEATHER DRIFT
    # ========================================================

    elif label == "WEATHER_DRIFT":

        humidity_boost = max(
            humidity_factor,
            0.0,
        )

        amplitude = (
            rng.uniform(
                0.07,
                0.24,
            )
            * (
                0.65
                + 0.70 * humidity_boost
            )
        )

        center = (
            onset
            + rng.uniform(
                0.15,
                0.45,
            )
        )

        width = rng.uniform(
            0.45,
            0.85,
        )

        rise = sigmoid(
            t,
            center,
            width,
        )

        recovery = sigmoid(
            t,
            center + rng.uniform(
                0.8,
                1.6,
            ),
            rng.uniform(
                0.45,
                0.80,
            ),
        )

        response = np.clip(
            rise
            - rng.uniform(
                0.10,
                0.30,
            ) * recovery,
            0.0,
            None,
        )

        weights = rng.uniform(
            0.65,
            1.00,
            4,
        )

        sensors += (
            amplitude
            * response[:, None]
            * weights
        )

        sensors += (
            humidity_factor
            * correlated_noise(
                N_SAMPLES,
                0.006,
                alpha=0.98,
            )[:, None]
        )

    # ========================================================
    # ALCOHOL / SANITIZER
    # ========================================================

    elif label == "ALCOHOL_SANITIZER":

        amplitude = rng.uniform(
            0.22,
            0.95,
        )

        center = (
            onset
            + rng.uniform(
                0.00,
                0.35,
            )
        )

        width = rng.uniform(
            0.08,
            0.28,
        )

        rise = sigmoid(
            t,
            center,
            width,
        )

        recovery = sigmoid(
            t,
            center + rng.uniform(
                0.45,
                1.20,
            ),
            rng.uniform(
                0.25,
                0.60,
            ),
        )

        response = np.clip(
            rise
            - rng.uniform(
                0.15,
                0.45,
            ) * recovery,
            0.0,
            None,
        )

        # Strong MQ-3 response.
        sensors[:, 1] += (
            amplitude
            * response
        )

        # Real sensors are not perfectly selective.
        sensors[:, 0] += (
            rng.uniform(
                0.015,
                0.12,
            )
            * response
        )

        sensors[:, 2] += (
            rng.uniform(
                0.015,
                0.14,
            )
            * response
        )

        sensors[:, 3] += (
            rng.uniform(
                0.00,
                0.07,
            )
            * response
        )

        # Weak events create overlap.
        if rng.random() < 0.25:

            sensors[:, 1] *= rng.uniform(
                0.65,
                0.85,
            )

    # ========================================================
    # EXPLOSIVE PROXY
    # ========================================================

    elif label == "EXPLOSIVE_PROXY":

        amplitude = rng.uniform(
            0.16,
            0.58,
        )

        center = (
            onset
            + rng.uniform(
                -0.10,
                0.30,
            )
        )

        width = rng.uniform(
            0.16,
            0.42,
        )

        rise = sigmoid(
            t,
            center,
            width,
        )

        recovery = sigmoid(
            t,
            center + rng.uniform(
                0.60,
                1.50,
            ),
            rng.uniform(
                0.35,
                0.75,
            ),
        )

        response = np.clip(
            rise
            - rng.uniform(
                0.10,
                0.35,
            ) * recovery,
            0.0,
            None,
        )

        # SEN0568 dominant.
        sensors[:, 3] += (
            amplitude
            * rng.uniform(
                0.75,
                1.15,
            )
            * response
        )

        # MQ-135 correlated.
        sensors[:, 2] += (
            amplitude
            * rng.uniform(
                0.60,
                1.00,
            )
            * response
        )

        # Moderate MQ-2.
        sensors[:, 0] += (
            amplitude
            * rng.uniform(
                0.20,
                0.55,
            )
            * response
        )

        # Small MQ-3 cross-response.
        sensors[:, 1] += (
            amplitude
            * rng.uniform(
                0.03,
                0.20,
            )
            * response
        )

        temperature = np.clip(
            temperature + rng.normal(
                0,
                0.8,
            ),
            10.0,
            40.0,
        )

        humidity = np.clip(
            humidity + rng.normal(
                0,
                2.5,
            ),
            20.0,
            90.0,
        )

    # ========================================================
    # NARCOTIC PROXY
    # ========================================================

    elif label == "NARCOTIC_PROXY":

        amplitude = rng.uniform(
            0.14,
            0.55,
        )

        center = (
            onset
            + rng.uniform(
                -0.10,
                0.35,
            )
        )

        width = rng.uniform(
            0.17,
            0.45,
        )

        rise = sigmoid(
            t,
            center,
            width,
        )

        recovery = sigmoid(
            t,
            center + rng.uniform(
                0.65,
                1.50,
            ),
            rng.uniform(
                0.35,
                0.80,
            ),
        )

        response = np.clip(
            rise
            - rng.uniform(
                0.10,
                0.35,
            ) * recovery,
            0.0,
            None,
        )

        # MQ-3 dominant.
        sensors[:, 1] += (
            amplitude
            * rng.uniform(
                0.65,
                1.10,
            )
            * response
        )

        # MQ-135 correlated.
        sensors[:, 2] += (
            amplitude
            * rng.uniform(
                0.60,
                1.05,
            )
            * response
        )

        # Moderate MQ-2.
        sensors[:, 0] += (
            amplitude
            * rng.uniform(
                0.20,
                0.60,
            )
            * response
        )

        # Small SEN0568 response.
        sensors[:, 3] += (
            amplitude
            * rng.uniform(
                0.03,
                0.25,
            )
            * response
        )

        # Additional cross-sensitivity.
        if rng.random() < 0.25:

            sensors[:, 0] += (
                rng.uniform(
                    0.04,
                    0.16,
                )
                * response
            )

        if rng.random() < 0.20:

            sensors[:, 3] += (
                rng.uniform(
                    0.03,
                    0.12,
                )
                * response
            )

    # ========================================================
    # ADC-LIKE QUANTIZATION
    # ========================================================

    sensors = clip_voltage(sensors)

    adc_levels = rng.choice(
        [4095, 4096, 8191],
        p=[0.55, 0.25, 0.20],
    )

    sensors = (
        np.round(
            sensors / 3.3 * adc_levels
        )
        / adc_levels
        * 3.3
    )

    sensors = clip_voltage(sensors)

    # ========================================================
    # dV/dt_max
    # ========================================================

    # First 300 ms:
    # 0.0 -> 0.1 -> 0.2 -> 0.3 seconds.

    early_slopes = (
        np.diff(
            sensors[:4],
            axis=0,
        )
        / DT
    )

    dVdt_max = np.max(
        np.abs(early_slopes)
    )

    # ========================================================
    # WINDOW FEATURE VALUES
    # ========================================================

    # Average final 500 ms to reduce single-sample noise.
    final_values = np.mean(
        sensors[-5:],
        axis=0,
    )

    temperature = np.clip(
        temperature,
        10.0,
        40.0,
    )

    humidity = np.clip(
        humidity,
        20.0,
        90.0,
    )

    return [
        final_values[0],
        final_values[1],
        final_values[2],
        final_values[3],
        dVdt_max,
        temperature,
        humidity,
    ]


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(samples_per_class):

    rows = []

    for label in CLASSES:

        print(
            f"Generating {samples_per_class:,} samples: {label}"
        )

        for _ in range(samples_per_class):

            features = generate_window(
                label
            )

            rows.append(
                features + [label]
            )

    df = pd.DataFrame(
        rows,
        columns=FEATURES + ["label"],
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SENTRY SYNTHETIC SENSOR DATASET")
    print("=" * 72)

    print(f"Random seed: {SEED}")
    print(f"Samples/window: {N_SAMPLES}")
    print("Sampling interval: 100 ms")
    print("Window duration: 3.0 seconds")
    print()

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    print("Generating TRAINING dataset...")
    print()

    train_df = generate_dataset(
        TRAIN_PER_CLASS
    )

    # --------------------------------------------------------
    # FINAL TEST DATA
    # --------------------------------------------------------

    print()
    print("Generating FINAL TEST dataset...")
    print()

    test_df = generate_dataset(
        TEST_PER_CLASS
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DATASET GENERATION COMPLETE")
    print("=" * 72)

    print()
    print(f"Training samples: {len(train_df):,}")
    print(f"Final test samples: {len(test_df):,}")
    print(
        f"Total samples: "
        f"{len(train_df) + len(test_df):,}"
    )

    print()
    print("Training class distribution:")
    print(
        train_df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Final test class distribution:")
    print(
        test_df["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Training feature statistics:")
    print(
        train_df[FEATURES]
        .describe()
        .round(4)
        .to_string()
    )

    print()
    print("Training class means:")
    print(
        train_df
        .groupby("label")[FEATURES]
        .mean()
        .round(4)
        .to_string()
    )

    print()
    print("Training class standard deviations:")
    print(
        train_df
        .groupby("label")[FEATURES]
        .std()
        .round(4)
        .to_string()
    )

    print()
    print("Saved:")
    print(TRAIN_FILE)
    print(TEST_FILE)

    print()
    print("IMPORTANT:")
    print(
        "Synthetic behavior-inspired data only. "
        "Not laboratory-calibrated real sensor data."
    )


if __name__ == "__main__":
    main()