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

REAL_BASELINES = {
    "MQ2": 3069.0,
    "MQ3": 530.33,
    "MQ135": 1043.27
}

MODEL_BASELINES = {
    "MQ2": 0.82,
    "MQ3": 0.74,
    "MQ135": 0.91
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


# ============================================================
# CONVERSION
# ============================================================

def adc_from_model(model_value, sensor):

    return (
        model_value
        / MODEL_BASELINES[sensor]
        * REAL_BASELINES[sensor]
    )


# ============================================================
# COMPENSATION
# ============================================================

def compensate(
    value,
    sensor,
    temperature,
    humidity
):

    return (
        value
        - ALPHA_H[sensor] * (humidity - REFERENCE_H)
        - ALPHA_T[sensor] * (temperature - REFERENCE_T)
    )


# ============================================================
# CREATE ONE WINDOW
# ============================================================

def make_window(label):

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    temperature = rng.normal(
        25.0,
        1.8
    )

    humidity = np.clip(
        rng.normal(
            50.0,
            6.0
        ),
        30,
        75
    )


    # --------------------------------------------------------
    # NORMAL BASELINE
    # --------------------------------------------------------

    base = np.array([
        0.82,   # MQ2
        0.74,   # MQ3
        0.91,   # MQ135
        0.34    # SEN0567
    ])


    # ========================================================
    # CLASS SIGNATURE
    # ========================================================

    if label == "SAFE":

        # Very tight normal-air distribution.
        target = base.copy()

        noise = np.array([
            0.008,
            0.008,
            0.010,
            0.005
        ])

        event_strength = 0.0
        onset = 30


    elif label == "WEATHER":

        # Weather changes environment rather than producing
        # a strong chemical event.

        env_strength = rng.uniform(
            0.8,
            1.2
        )

        weather_pattern = np.array([
            0.025,
            0.018,
            0.035,
            0.012
        ])

        target = (
            base
            + weather_pattern * env_strength
        )

        noise = np.array([
            0.012,
            0.012,
            0.015,
            0.007
        ])

        event_strength = 0.0
        onset = 30


    elif label == "ALCOHOL":

        target = (
            base
            + np.array([
                0.11,
                0.34,
                0.16,
                0.04
            ])
        )

        noise = np.array([
            0.018,
            0.018,
            0.020,
            0.010
        ])

        event_strength = rng.uniform(
            0.75,
            1.05
        )

        onset = rng.integers(
            7,
            14
        )


    elif label == "EXPLOSIVE":

        target = (
            base
            + np.array([
                0.34,
                0.14,
                0.30,
                0.12
            ])
        )

        noise = np.array([
            0.020,
            0.020,
            0.022,
            0.012
        ])

        event_strength = rng.uniform(
            0.85,
            1.15
        )

        onset = rng.integers(
            3,
            9
        )


    elif label == "NARCOTIC":

        target = (
            base
            + np.array([
                0.18,
                0.10,
                0.27,
                0.20
            ])
        )

        noise = np.array([
            0.020,
            0.020,
            0.022,
            0.012
        ])

        event_strength = rng.uniform(
            0.50,
            0.90
        )

        onset = rng.integers(
            7,
            16
        )


    else:

        raise ValueError(
            f"Unknown class: {label}"
        )


    # ========================================================
    # GENERATE 30-SAMPLE TRAJECTORY
    # ========================================================

    trajectory = np.zeros(
        (N_SAMPLES, 4)
    )


    for i in range(N_SAMPLES):

        if (
            i >= onset
            and event_strength > 0
        ):

            progress = min(
                1.0,
                (i - onset + 1) / 7.0
            )

            event = (
                target - base
            ) * progress * event_strength

            value = (
                base + event
            )

        else:

            value = base.copy()


        # Sensor noise
        value = (
            value
            + rng.normal(
                0,
                noise
            )
        )


        trajectory[i] = value


    # ========================================================
    # CONVERT TO HARDWARE-LIKE VALUES
    # ========================================================

    mq2_adc = np.clip(
        adc_from_model(
            trajectory[:, 0],
            "MQ2"
        )
        + rng.normal(0, 8, N_SAMPLES),
        0,
        4095
    )

    mq3_adc = np.clip(
        adc_from_model(
            trajectory[:, 1],
            "MQ3"
        )
        + rng.normal(0, 3, N_SAMPLES),
        0,
        4095
    )

    mq135_adc = np.clip(
        adc_from_model(
            trajectory[:, 2],
            "MQ135"
        )
        + rng.normal(0, 5, N_SAMPLES),
        0,
        4095
    )

    sen0567_v = np.clip(
        trajectory[:, 3]
        + rng.normal(
            0,
            0.004,
            N_SAMPLES
        ),
        0,
        3.3
    )


    # ========================================================
    # ENVIRONMENT SERIES
    # ========================================================

    temperature_series = (
        temperature
        + rng.normal(
            0,
            0.10,
            N_SAMPLES
        )
    )

    humidity_series = (
        humidity
        + rng.normal(
            0,
            0.30,
            N_SAMPLES
        )
    )


    # ========================================================
    # CONVERT BACK TO MODEL SPACE
    # ========================================================

    mq2_model = (
        mq2_adc
        / REAL_BASELINES["MQ2"]
        * MODEL_BASELINES["MQ2"]
    )

    mq3_model = (
        mq3_adc
        / REAL_BASELINES["MQ3"]
        * MODEL_BASELINES["MQ3"]
    )

    mq135_model = (
        mq135_adc
        / REAL_BASELINES["MQ135"]
        * MODEL_BASELINES["MQ135"]
    )


    # ========================================================
    # COMPENSATION
    # ========================================================

    mq2_comp = compensate(
        mq2_model,
        "MQ2",
        temperature_series,
        humidity_series
    )

    mq3_comp = compensate(
        mq3_model,
        "MQ3",
        temperature_series,
        humidity_series
    )

    mq135_comp = compensate(
        mq135_model,
        "MQ135",
        temperature_series,
        humidity_series
    )

    sen_comp = compensate(
        sen0567_v,
        "SEN0567",
        temperature_series,
        humidity_series
    )


    # ========================================================
    # dV/dt
    # ========================================================

    early = np.column_stack([
        mq2_comp[:4],
        mq3_comp[:4],
        mq135_comp[:4],
        sen_comp[:4]
    ])

    slopes = (
        np.diff(
            early,
            axis=0
        ) / DT
    )

    dVdt_max = float(
        np.max(
            np.abs(slopes)
        )
    )


    # ========================================================
    # FINAL 5-READING AVERAGE
    # ========================================================

    final_slice = slice(
        N_SAMPLES - 5,
        N_SAMPLES
    )


    return {

        "VMQ2":
            float(
                np.mean(
                    mq2_comp[final_slice]
                )
            ),

        "VMQ3":
            float(
                np.mean(
                    mq3_comp[final_slice]
                )
            ),

        "VMQ135":
            float(
                np.mean(
                    mq135_comp[final_slice]
                )
            ),

        "VSEN0567":
            float(
                np.mean(
                    sen_comp[final_slice]
                )
            ),

        "dVdt_max":
            dVdt_max,

        "temperature":
            float(
                np.mean(
                    temperature_series[
                        final_slice
                    ]
                )
            ),

        "humidity":
            float(
                np.mean(
                    humidity_series[
                        final_slice
                    ]
                )
            ),

        "label":
            label
    }


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    samples_per_class,
    path
):

    rows = []

    for label in CLASSES:

        print(
            f"Generating {label}..."
        )

        for _ in range(
            samples_per_class
        ):

            rows.append(
                make_window(label)
            )


    df = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )

    print()
    print(
        f"Saved: {path}"
    )

    print(
        f"Shape: {df.shape}"
    )

    print(
        df["label"].value_counts()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_dataset(
        TRAIN_SAMPLES_PER_CLASS,
        "data/raw/sentry_synthetic_dataset.csv"
    )

    generate_dataset(
        TEST_SAMPLES_PER_CLASS,
        "data/raw/sentry_synthetic_test.csv"
    )