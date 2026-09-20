FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
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

SAMPLE_INTERVAL_MS = 100
SAMPLE_INTERVAL_S = SAMPLE_INTERVAL_MS / 1000.0
WINDOW_SECONDS = 3
SAMPLES_PER_WINDOW = 30
N_SAMPLES = SAMPLES_PER_WINDOW

H_REF = 50.0
T_REF = 25.0

# Model-space baselines for the 3 MQ sensors.
# These correspond to the operating region used by the
# current 6-feature synthetic dataset.
MODEL_BASELINES = {
    "MQ2": 0.48,
    "MQ3": 2.68,
    "MQ135": 0.42,
}

# Approximate ESP32 clean-air raw baselines.
# Demonstration calibration values, not laboratory calibration constants.
REAL_BASELINES = {
    "MQ2": 1800.0,
    "MQ3": 1920.0,
    "MQ135": 480.0,
}

# Prototype environmental compensation coefficients
# for MQ2, MQ3 and MQ135 respectively.
COMPENSATION_ALPHA_H = [
    0.0018,  # MQ2
    0.0014,  # MQ3
    0.0022,  # MQ135
]

COMPENSATION_ALPHA_T = [
    0.0008,  # MQ2
    0.0007,  # MQ3
    0.0009,  # MQ135
]