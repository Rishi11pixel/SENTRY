FEATURES = [
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

SAMPLE_INTERVAL_MS = 100
SAMPLE_INTERVAL_S = SAMPLE_INTERVAL_MS / 1000.0
WINDOW_SECONDS = 3
SAMPLES_PER_WINDOW = 30
N_SAMPLES = SAMPLES_PER_WINDOW

H_REF = 50.0
T_REF = 25.0

# These are now centered on the real operating region observed from
# the current ESP32 + sensor setup, rather than the old synthetic
# 0.x SEN0567 / MQ3 operating region.
MODEL_BASELINES = {
    "MQ2": 0.48,
    "MQ3": 2.68,
    "MQ135": 0.42,
    "SEN0567": 1.72,
}

# These are approximate current ESP32 clean-air raw baselines.
# Recheck after hardware warm-up; they are demonstration calibration
# values, not laboratory calibration constants.
REAL_BASELINES = {
    "MQ2": 1800.0,
    "MQ3": 1920.0,
    "MQ135": 480.0,
}

# Prototype compensation coefficients used by the synthetic simulator
# and server. They are not laboratory calibration constants.
COMPENSATION_ALPHA_H = [0.0018, 0.0014, 0.0022, 0.0016]
COMPENSATION_ALPHA_T = [0.0008, 0.0007, 0.0009, 0.0008]
