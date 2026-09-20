import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

TRAIN_SAMPLES_PER_CLASS = 12000
TEST_SAMPLES_PER_CLASS = 2400

CLASSES = [
    "SAFE",
    "WEATHER",
    "ALCOHOL",
    "EXPLOSIVE",
    "NARCOTIC"
]

# ------------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------------
# This is a demonstration-oriented synthetic dataset.
#
# SAFE and WEATHER are anchored to the real ESP32 feature values
# supplied during testing.
#
# EXPLOSIVE and NARCOTIC are proxy signatures derived from the
# observed Move Gel / Harpic percentage responses. They are NOT
# chemically validated explosive/narcotic signatures.
#
# ALCOHOL is an additional synthetic proxy pattern placed between
# the observed response families. It is not a real alcohol signature.
# ------------------------------------------------------------------

MODEL_BASELINES = {
    "MQ2": 0.48,
    "MQ3": 2.68,
    "MQ135": 0.42,
    "SEN0567": 1.72
}

REAL_BASELINES = {
    "MQ2": 1800.0,
    "MQ3": 1920.0,
    "MQ135": 480.0
}

ALPHA_H = {
    "MQ2": 0.0018,
    "MQ3": 0.0014,
    "MQ135": 0.0022,
    "SEN0567": 0.0016
}

ALPHA_T = {
    "MQ2": 0.0008,
    "MQ3": 0.0007,
    "MQ135": 0.0009,
    "SEN0567": 0.0008
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


def compensate(value, sensor, temperature, humidity):
    return (
        value
        - ALPHA_H[sensor] * (humidity - REFERENCE_H)
        - ALPHA_T[sensor] * (temperature - REFERENCE_T)
    )


def make_window(label):

    # Current real clean-air operating point from the supplied ESP32
    # logs, approximately:
    # [0.48, 2.68, 0.42, 1.72]
    clean = np.array([
        0.48,
        2.68,
        0.42,
        1.72
    ], dtype=float)

    # --------------------------------------------------------------
    # Observed percentage responses from the user's real tests
    #
    # Move Gel:
    # MQ2 -3.2%, MQ3 -0.1%, MQ135 -4.7%, SEN0567 -0.3%
    #
    # Harpic:
    # MQ2 -2.62%, MQ3 +0.03%, MQ135 -5.39%, SEN0567 +0.61%
    # --------------------------------------------------------------

    move_response = np.array([
        -0.032,
        -0.001,
        -0.047,
        -0.003
    ])

    harpic_response = np.array([
        -0.0262,
         0.0003,
        -0.0539,
         0.0061
    ])

    temperature = np.clip(
        rng.normal(30.5, 1.2),
        26.0,
        35.0
    )

    humidity = np.clip(
        rng.normal(65.0, 5.0),
        45.0,
        80.0
    )

    # --------------------------------------------------------------
    # Class trajectory definitions
    # --------------------------------------------------------------

    if label == "SAFE":

        target = clean.copy()
        event_delta = np.zeros(4)
        noise = np.array([0.010, 0.018, 0.012, 0.006])
        onset = 30
        event_strength = 0.0
        recovery = False

    elif label == "WEATHER":

        # Anchored to the actual gradual drift seen in the supplied
        # "weather" outputs: sensors slowly decrease while T/H drift.
        weather_delta = np.array([
            -0.060,   # MQ2
            -0.050,   # MQ3
             0.005,  # MQ135
            -0.045   # SEN0567
        ])

        target = clean + weather_delta
        event_delta = weather_delta
        noise = np.array([0.012, 0.020, 0.014, 0.007])
        onset = 0
        event_strength = rng.uniform(0.75, 1.15)
        recovery = False

    elif label == "EXPLOSIVE":

        # Move Gel proxy: preserve the observed direction but make
        # the synthetic event large enough to be distinguishable.
        # This is a sensor-pattern proxy, NOT an explosive signature.
        event_delta = clean * move_response * rng.uniform(2.0, 3.0)

        # Add a sharper onset because the model uses dV/dt.
        target = clean + event_delta
        noise = np.array([0.014, 0.020, 0.016, 0.008])
        onset = rng.integers(3, 8)
        event_strength = rng.uniform(0.75, 1.0)
        recovery = True

    elif label == "NARCOTIC":

        # Harpic proxy: preserve observed response direction.
        # This is NOT a narcotic chemical signature.
        event_delta = clean * harpic_response * rng.uniform(2.0, 3.0)

        target = clean + event_delta
        noise = np.array([0.014, 0.020, 0.016, 0.008])
        onset = rng.integers(7, 14)
        event_strength = rng.uniform(0.70, 1.0)
        recovery = True

    elif label == "ALCOHOL":

        # Synthetic intermediate proxy. It deliberately uses a
        # different multi-sensor direction rather than pretending
        # Harpic/Move Gel are alcohol.
        event_delta = np.array([
            -0.030,
             0.055,
            -0.075,
             0.030
        ])

        target = clean + event_delta
        noise = np.array([0.014, 0.020, 0.016, 0.008])
        onset = rng.integers(6, 13)
        event_strength = rng.uniform(0.70, 1.0)
        recovery = True

    else:
        raise ValueError(f"Unknown class: {label}")

    trajectory = np.zeros((N_SAMPLES, 4), dtype=float)

    for i in range(N_SAMPLES):

        if label == "SAFE":
            value = clean.copy()

        elif label == "WEATHER":
            # Slow environmental drift across the window.
            progress = i / (N_SAMPLES - 1)
            value = clean + event_delta * progress * event_strength

        elif i >= onset:
            progress = min(
                1.0,
                (i - onset + 1) / 6.0
            )

            event = event_delta * progress * event_strength
            value = clean + event

            # Partial recovery near the end for exposure proxies.
            if recovery and i >= 24:
                recovery_progress = (i - 24) / 5.0
                value = value - event * 0.55 * recovery_progress

        else:
            value = clean.copy()

        value = value + rng.normal(0, noise)
        trajectory[i] = value

    # --------------------------------------------------------------
    # Convert to hardware-like values, then run the SAME conversion
    # and compensation path as the server.
    # --------------------------------------------------------------

    mq2_adc = np.clip(
        adc_from_model(trajectory[:, 0], "MQ2")
        + rng.normal(0, 10, N_SAMPLES),
        0,
        4095
    )

    mq3_adc = np.clip(
        adc_from_model(trajectory[:, 1], "MQ3")
        + rng.normal(0, 5, N_SAMPLES),
        0,
        4095
    )

    mq135_adc = np.clip(
        adc_from_model(trajectory[:, 2], "MQ135")
        + rng.normal(0, 7, N_SAMPLES),
        0,
        4095
    )

    sen0567_v = np.clip(
        trajectory[:, 3]
        + rng.normal(0, 0.006, N_SAMPLES),
        0.0,
        3.3
    )

    temperature_series = (
        temperature
        + rng.normal(0, 0.10, N_SAMPLES)
    )

    humidity_series = (
        humidity
        + rng.normal(0, 0.30, N_SAMPLES)
    )

    mq2_model = (
        mq2_adc / REAL_BASELINES["MQ2"]
        * MODEL_BASELINES["MQ2"]
    )

    mq3_model = (
        mq3_adc / REAL_BASELINES["MQ3"]
        * MODEL_BASELINES["MQ3"]
    )

    mq135_model = (
        mq135_adc / REAL_BASELINES["MQ135"]
        * MODEL_BASELINES["MQ135"]
    )

    mq2_comp = compensate(
        mq2_model, "MQ2",
        temperature_series, humidity_series
    )

    mq3_comp = compensate(
        mq3_model, "MQ3",
        temperature_series, humidity_series
    )

    mq135_comp = compensate(
        mq135_model, "MQ135",
        temperature_series, humidity_series
    )

    sen_comp = compensate(
        sen0567_v, "SEN0567",
        temperature_series, humidity_series
    )

    early = np.column_stack([
        mq2_comp[:4],
        mq3_comp[:4],
        mq135_comp[:4],
        sen_comp[:4]
    ])

    slopes = np.diff(early, axis=0) / DT

    dVdt_max = float(
        np.max(np.abs(slopes))
    )

    final_slice = slice(-5, None)

    return {
        "VMQ2": float(np.mean(mq2_comp[final_slice])),
        "VMQ3": float(np.mean(mq3_comp[final_slice])),
        "VMQ135": float(np.mean(mq135_comp[final_slice])),
        "VSEN0567": float(np.mean(sen_comp[final_slice])),
        "dVdt_max": dVdt_max,
        "temperature": float(
            np.mean(temperature_series[final_slice])
        ),
        "humidity": float(
            np.mean(humidity_series[final_slice])
        ),
        "label": label
    }


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
    print(f"Saved: {path}")
    print(f"Shape: {df.shape}")
    print(df["label"].value_counts())


if __name__ == "__main__":

    generate_dataset(
        TRAIN_SAMPLES_PER_CLASS,
        "data/raw/sentry_synthetic_dataset.csv"
    )

    generate_dataset(
        TEST_SAMPLES_PER_CLASS,
        "data/raw/sentry_synthetic_test.csv"
    )