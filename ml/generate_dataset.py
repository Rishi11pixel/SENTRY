import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

TRAIN_SAMPLES_PER_CLASS = 12000
TEST_SAMPLES_PER_CLASS = 2400

CLASSES = ["SAFE", "WEATHER", "ALCOHOL", "EXPLOSIVE", "NARCOTIC"]

# Actual ESP32 MQ ADC baselines
REAL_BASELINES = {
    "MQ2": 3069.0,
    "MQ3": 530.33,
    "MQ135": 1043.27,
}

# Model-space baselines used by the existing pipeline
MODEL_BASELINES = {
    "MQ2": 0.82,
    "MQ3": 0.74,
    "MQ135": 0.91,
}

# Prototype environmental compensation coefficients
ALPHA_H = {
    "MQ2": 0.0018,
    "MQ3": 0.0014,
    "MQ135": 0.0022,
    "SEN0567": 0.0016,
}

ALPHA_T = {
    "MQ2": 0.0008,
    "MQ3": 0.0007,
    "MQ135": 0.0009,
    "SEN0567": 0.0008,
}

REFERENCE_H = 50.0
REFERENCE_T = 25.0

DT = 0.1
N_SAMPLES = 30


def adc_from_model(model_value, sensor):
    return (
        model_value
        / MODEL_BASELINES[sensor]
        * REAL_BASELINES[sensor]
    )


def compensate(v, sensor, temperature, humidity):
    return (
        v
        - ALPHA_H[sensor] * (humidity - REFERENCE_H)
        - ALPHA_T[sensor] * (temperature - REFERENCE_T)
    )


def make_window(label):
    """
    Generate a synthetic 30-sample window in the same physical
    representation used by the ESP32:

        MQ2    -> ADC
        MQ3    -> ADC
        MQ135  -> ADC
        SEN0567 -> volts
        T      -> Celsius
        H      -> relative humidity
    """

    # Environmental conditions
    temperature = rng.normal(25.0, 2.0)
    humidity = np.clip(rng.normal(50.0, 8.0), 20.0, 85.0)

    # Base model-space sensor levels.
    # These are deliberately centered around the actual model baselines.
    base = np.array([
        MODEL_BASELINES["MQ2"],
        MODEL_BASELINES["MQ3"],
        MODEL_BASELINES["MQ135"],
        0.34,
    ])

    if label == "SAFE":
        target = base + rng.normal(0, [0.025, 0.025, 0.030, 0.015])

        event_strength = 0.0
        onset = 0

    elif label == "WEATHER":
        # Environmental shift rather than a chemical event.
        env = (
            0.004 * (humidity - 50.0)
            + 0.006 * (temperature - 25.0)
        )

        target = base.copy()
        target += env * np.array([0.8, 0.6, 1.0, 0.7])
        target += rng.normal(0, [0.035, 0.035, 0.040, 0.020])

        event_strength = 0.0
        onset = 0

    elif label == "ALCOHOL":
        target = base + np.array([
            0.10,
            0.34,
            0.16,
            0.04,
        ])

        target += rng.normal(0, [0.035, 0.035, 0.040, 0.020])

        event_strength = rng.uniform(0.6, 1.0)
        onset = rng.integers(4, 15)

    elif label == "EXPLOSIVE":
        target = base + np.array([
            0.34,
            0.14,
            0.30,
            0.12,
        ])

        target += rng.normal(0, [0.040, 0.040, 0.045, 0.025])

        event_strength = rng.uniform(0.8, 1.2)
        onset = rng.integers(3, 10)

    elif label == "NARCOTIC":
        target = base + np.array([
            0.18,
            0.10,
            0.27,
            0.20,
        ])

        target += rng.normal(0, [0.040, 0.040, 0.045, 0.025])

        event_strength = rng.uniform(0.4, 0.85)
        onset = rng.integers(6, 17)

    else:
        raise ValueError(label)

    # Build 30-sample model-space trajectories first.
    trajectory = np.zeros((N_SAMPLES, 4))

    for i in range(N_SAMPLES):

        # Small natural sensor drift
        drift = rng.normal(
            0,
            [0.003, 0.003, 0.004, 0.002]
        )

        # Smooth event transition
        if i >= onset and event_strength > 0:
            progress = min(
                1.0,
                (i - onset + 1) / 6.0
            )

            event = (
                target - base
            ) * progress * event_strength

            value = base + event
        else:
            value = base.copy()

        value += drift
        value += rng.normal(
            0,
            [0.012, 0.012, 0.015, 0.008]
        )

        trajectory[i] = value

    # Convert MQ model-space values back into ESP32-like ADC readings.
    mq2_adc = adc_from_model(
        trajectory[:, 0],
        "MQ2"
    )

    mq3_adc = adc_from_model(
        trajectory[:, 1],
        "MQ3"
    )

    mq135_adc = adc_from_model(
        trajectory[:, 2],
        "MQ135"
    )

    # SEN0567 is already represented as voltage.
    sen0567_v = trajectory[:, 3]

    # Add realistic ADC noise / quantisation.
    mq2_adc = np.clip(
        mq2_adc + rng.normal(0, 12, N_SAMPLES),
        0,
        4095
    )

    mq3_adc = np.clip(
        mq3_adc + rng.normal(0, 4, N_SAMPLES),
        0,
        4095
    )

    mq135_adc = np.clip(
        mq135_adc + rng.normal(0, 7, N_SAMPLES),
        0,
        4095
    )

    sen0567_v = np.clip(
        sen0567_v + rng.normal(0, 0.006, N_SAMPLES),
        0.0,
        3.3
    )

    # Environmental readings fluctuate slightly during the window.
    temperature_series = (
        temperature
        + rng.normal(0, 0.15, N_SAMPLES)
    )

    humidity_series = (
        humidity
        + rng.normal(0, 0.5, N_SAMPLES)
    )

    # Convert MQ ADC back to model-space exactly as deployment will.
    mq2_model = (
        mq2_adc / REAL_BASELINES["MQ2"]
    ) * MODEL_BASELINES["MQ2"]

    mq3_model = (
        mq3_adc / REAL_BASELINES["MQ3"]
    ) * MODEL_BASELINES["MQ3"]

    mq135_model = (
        mq135_adc / REAL_BASELINES["MQ135"]
    ) * MODEL_BASELINES["MQ135"]

    # Environmental compensation
    mq2_comp = compensate(
        mq2_model,
        "MQ2",
        temperature_series,
        humidity_series,
    )

    mq3_comp = compensate(
        mq3_model,
        "MQ3",
        temperature_series,
        humidity_series,
    )

    mq135_comp = compensate(
        mq135_model,
        "MQ135",
        temperature_series,
        humidity_series,
    )

    sen_comp = compensate(
        sen0567_v,
        "SEN0567",
        temperature_series,
        humidity_series,
    )

    # Early dV/dt: first four samples.
    early = np.column_stack([
        mq2_comp[:4],
        mq3_comp[:4],
        mq135_comp[:4],
        sen_comp[:4],
    ])

    slopes = np.diff(early, axis=0) / DT
    dVdt_max = float(np.max(np.abs(slopes)))

    # Final 5-sample averages.
    final_start = N_SAMPLES - 5

    features = {
        "VMQ2": float(np.mean(mq2_comp[final_start:])),
        "VMQ3": float(np.mean(mq3_comp[final_start:])),
        "VMQ135": float(np.mean(mq135_comp[final_start:])),
        "VSEN0567": float(np.mean(sen_comp[final_start:])),
        "dVdt_max": dVdt_max,
        "temperature": float(
            np.mean(temperature_series[final_start:])
        ),
        "humidity": float(
            np.mean(humidity_series[final_start:])
        ),
        "label": label,
    }

    return features


def generate_dataset(samples_per_class, path):
    rows = []

    for label in CLASSES:
        print(f"Generating {label}...")

        for _ in range(samples_per_class):
            rows.append(make_window(label))

    df = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df.to_csv(path, index=False)

    print()
    print(path)
    print("Shape:", df.shape)
    print(df["label"].value_counts())
    print()


if __name__ == "__main__":

    generate_dataset(
        TRAIN_SAMPLES_PER_CLASS,
        "data/raw/sentry_synthetic_dataset.csv",
    )

    generate_dataset(
        TEST_SAMPLES_PER_CLASS,
        "data/raw/sentry_synthetic_test.csv",
    )