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

# Prototype coefficients for synthetic simulation only; not laboratory calibration.
COMPENSATION_ALPHA_H = [0.0018, 0.0014, 0.0022, 0.0016]
COMPENSATION_ALPHA_T = [0.0008, 0.0007, 0.0009, 0.0008]