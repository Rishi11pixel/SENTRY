from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CLASSES,
    COMPENSATION_ALPHA_H,
    COMPENSATION_ALPHA_T,
    FEATURES,
    H_REF,
    MODEL_BASELINES,
    N_SAMPLES,
    SAMPLE_INTERVAL_S,
    T_REF,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
TRAIN_FILE = OUTPUT_DIR / "sentry_synthetic_dataset.csv"
TEST_FILE = OUTPUT_DIR / "sentry_synthetic_test.csv"
TRAIN_PER_CLASS = 12_000
TEST_PER_CLASS = 2_400
TRAIN_SEED = 87
TEST_SEED = 1807
SENSOR_NAMES = ["VMQ2", "VMQ3", "VMQ135", "VSEN0567"]
BASELINE = np.array([MODEL_BASELINES[sensor] for sensor in SENSOR_NAMES], dtype=np.float32)
NOISE = np.array([0.012, 0.014, 0.013, 0.012], dtype=np.float32)


def correlated_noise(rng, n, sigma, alpha=0.90):
    raw = rng.normal(0.0, sigma, n)
    output = np.zeros(n)
    output[0] = raw[0]
    for i in range(1, n):
        output[i] = alpha * output[i - 1] + np.sqrt(1.0 - alpha**2) * raw[i]
    return output


def sigmoid(t, center, width):
    return 1.0 / (1.0 + np.exp(-(t - center) / width))


def generate_environment(label, rng, condition_override=None):
    if label == "WEATHER" or condition_override is not None:
        condition = condition_override or rng.choice(
            ["dry", "normal", "humid", "very_humid", "cold", "hot", "hot_humid", "hot_dry"],
            p=[0.10, 0.16, 0.14, 0.12, 0.10, 0.12, 0.14, 0.12],
        )
        ranges = {
            "dry": ((20.0, 30.0), (18.0, 30.0)),
            "normal": ((40.0, 60.0), (20.0, 30.0)),
            "humid": ((70.0, 80.0), (20.0, 30.0)),
            "very_humid": ((85.0, 90.0), (20.0, 30.0)),
            "cold": ((40.0, 70.0), (10.0, 18.0)),
            "hot": ((40.0, 70.0), (35.0, 40.0)),
            "hot_humid": ((70.0, 90.0), (35.0, 40.0)),
            "hot_dry": ((20.0, 35.0), (35.0, 40.0)),
        }
        humidity_range, temperature_range = ranges[condition]
        humidity = rng.uniform(*humidity_range)
        temperature = rng.uniform(*temperature_range)
        humidity_end = np.clip(humidity + rng.uniform(-5.0, 18.0), 20.0, 90.0)
        temperature_end = np.clip(temperature + rng.uniform(-3.0, 5.0), 10.0, 40.0)
        return temperature, humidity, temperature_end, humidity_end

    temperature = rng.uniform(20.0, 30.0)
    humidity = rng.uniform(40.0, 60.0)
    if label == "SAFE":
        if rng.random() < 0.65:
            humidity = rng.uniform(20.0, 90.0)
            temperature = rng.uniform(10.0, 40.0)
        else:
            humidity = rng.uniform(30.0, 75.0)
            temperature = rng.uniform(16.0, 34.0)
    return temperature, humidity, temperature, humidity


def event_response(label, t, rng):
    response = np.zeros((N_SAMPLES, 4), dtype=np.float32)
    onset = rng.uniform(0.08, 2.35)

    if label == "SAFE":
        if rng.random() < 0.15:
            sensor = rng.integers(0, 4)
            response[:, sensor] += rng.uniform(0.01, 0.06) * np.exp(
                -0.5 * ((t - rng.uniform(0.5, 2.5)) / rng.uniform(0.10, 0.30)) ** 2
            )
        return response

    if label == "WEATHER":
        drift = rng.uniform(0.03, 0.34)
        rise = sigmoid(t, onset + rng.uniform(0.15, 0.45), rng.uniform(0.45, 0.85))
        response += drift * rise[:, None] * rng.uniform(0.70, 1.10, 4)
        return response

    center = onset + rng.uniform(-0.10, 0.30)
    event_strength = rng.choice([0.05, 0.25, 1.00], p=[0.45, 0.35, 0.20])
    if label == "ALCOHOL":
        event = sigmoid(t, center, rng.uniform(0.05, 0.16))
        event *= 1.0 - rng.uniform(0.35, 0.75) * sigmoid(t, center + rng.uniform(0.35, 0.90), rng.uniform(0.18, 0.45))
        amplitude = rng.uniform(1.25, 2.15) * event_strength
        response[:, 1] += amplitude * event
        response[:, 0] += rng.uniform(0.05, 0.18) * amplitude * event
        response[:, 2] += rng.uniform(0.04, 0.20) * amplitude * event
        response[:, 3] += rng.uniform(0.02, 0.12) * amplitude * event
    elif label == "EXPLOSIVE":
        event = sigmoid(t, center, rng.uniform(0.10, 0.27))
        event *= 1.0 - rng.uniform(0.15, 0.55) * sigmoid(t, center + rng.uniform(0.55, 1.35), rng.uniform(0.30, 0.70))
        amplitude = rng.uniform(1.15, 1.70) * event_strength
        response[:, 3] += amplitude * rng.uniform(0.80, 1.10) * event
        response[:, 2] += amplitude * rng.uniform(0.75, 1.10) * event
        response[:, 0] += amplitude * rng.uniform(0.25, 0.48) * event
        response[:, 1] += amplitude * rng.uniform(0.05, 0.22) * event
    elif label == "NARCOTIC":
        event = sigmoid(t, center, rng.uniform(0.12, 0.30))
        event *= 1.0 - rng.uniform(0.15, 0.50) * sigmoid(t, center + rng.uniform(0.60, 1.45), rng.uniform(0.35, 0.80))
        amplitude = rng.uniform(1.35, 2.00) * event_strength
        response[:, 1] += amplitude * rng.uniform(0.78, 1.10) * event
        response[:, 2] += amplitude * rng.uniform(0.72, 1.08) * event
        response[:, 0] += amplitude * rng.uniform(0.24, 0.48) * event
        response[:, 3] += amplitude * rng.uniform(0.02, 0.12) * event
    return response


def generate_window(label, rng, apply_compensation=True, condition_override=None):
    t = np.arange(N_SAMPLES, dtype=np.float32) * SAMPLE_INTERVAL_S
    temperature, humidity, temperature_end, humidity_end = generate_environment(label, rng, condition_override)
    environmental_temperature = np.linspace(temperature, temperature_end, N_SAMPLES)
    environmental_humidity = np.linspace(humidity, humidity_end, N_SAMPLES)
    offsets = rng.normal(0.0, 0.018, 4)
    gains = rng.normal(1.0, 0.045, 4)
    raw = np.tile(BASELINE, (N_SAMPLES, 1)) * gains + offsets
    humidity_distortion = (environmental_humidity - H_REF)[:, None] * COMPENSATION_ALPHA_H
    temperature_distortion = (environmental_temperature - T_REF)[:, None] * COMPENSATION_ALPHA_T
    raw += humidity_distortion + temperature_distortion
    raw += correlated_noise(rng, N_SAMPLES, 0.010, 0.94)[:, None]
    raw += rng.normal(0.0, NOISE, size=(N_SAMPLES, 4))
    raw += event_response(label, t, rng)
    if label == "WEATHER":
        raw += ((environmental_humidity - H_REF) / 40.0)[:, None] * rng.uniform(0.005, 0.025, 4)
    adc_levels = rng.choice([4095, 8191], p=[0.7, 0.3])
    raw = np.clip(raw, 0.02, 3.25)
    raw = np.round(raw / 3.3 * adc_levels) / adc_levels * 3.3
    compensated = raw - humidity_distortion - temperature_distortion if apply_compensation else raw.copy()
    early_slopes = np.diff(compensated[:4], axis=0) / SAMPLE_INTERVAL_S
    d_v_dt_max = float(np.max(np.abs(early_slopes)))
    final_values = compensated[-5:].mean(axis=0)
    feature_values = [*final_values, d_v_dt_max, float(environmental_temperature[-1]), float(environmental_humidity[-1])]
    raw_values = [*raw[-5:].mean(axis=0), float(np.max(np.abs(np.diff(raw[:4], axis=0) / SAMPLE_INTERVAL_S)))]
    return feature_values, raw_values


def generate_dataset(samples_per_class, seed, apply_compensation=True, condition_override=None):
    rng = np.random.default_rng(seed)
    rows = []
    raw_rows = []
    for label in CLASSES:
        print(f"Generating {samples_per_class:,} samples: {label}")
        for _ in range(samples_per_class):
            features, raw_values = generate_window(label, rng, apply_compensation, condition_override)
            rows.append(features + [label])
            raw_rows.append(raw_values + [label])
    return pd.DataFrame(rows, columns=FEATURES + ["label"]), pd.DataFrame(
        raw_rows, columns=["VMQ2", "VMQ3", "VMQ135", "VSEN0567", "dVdt_max", "label"]
    )


def main():
    print("SENTRY synthetic sensor dataset generation")
    print(f"Training seed: {TRAIN_SEED}; final-test seed: {TEST_SEED}")
    print("Compensation equation: V_comp = V_raw - alpha_H*(humidity - 50.0) - alpha_T*(temperature - 25.0)")
    print("Prototype synthetic coefficients, not experimentally calibrated constants:")
    for sensor, alpha_h, alpha_t in zip(SENSOR_NAMES, COMPENSATION_ALPHA_H, COMPENSATION_ALPHA_T):
        print(f"  {sensor}: alpha_H={alpha_h:.6f}, alpha_T={alpha_t:.6f}")
    train_df, train_raw_df = generate_dataset(TRAIN_PER_CLASS, TRAIN_SEED)
    test_df, _ = generate_dataset(TEST_PER_CLASS, TEST_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    print("\nRaw sensor range statistics by class:")
    print(train_raw_df.groupby("label").agg(["min", "max", "mean"]).round(4).to_string())
    print("\nCompensated sensor range statistics by class:")
    print(train_df.groupby("label")[SENSOR_NAMES].agg(["min", "max", "mean"]).round(4).to_string())
    print(f"\nTraining samples: {len(train_df):,}; final test samples: {len(test_df):,}")
    print(train_df["label"].value_counts().sort_index().to_string())
    print("Synthetic data only; not laboratory-calibrated.")


if __name__ == "__main__":
    main()
