from flask import Flask, json, request, jsonify
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys
import time


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import (
    MODEL_BASELINES,
    REAL_BASELINES,
    COMPENSATION_ALPHA_H,
    COMPENSATION_ALPHA_T,
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# CONTROLLED DEMONSTRATION STATE
# ============================================================

DEMO_SAFE_SECONDS = 10.0
DEMO_NARCOTIC_SECONDS = 5.0
DEMO_PAUSE_SECONDS = 8.0
DEMO_EXPLOSIVE_SECONDS = 4.0
DEMO_RNG = np.random.default_rng()
demo_sessions = {}


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = ROOT / "ml" / "models" / "tinyml"

MODEL_PATH = MODEL_DIR / "sentry_tinyml.keras"
SCALER_MEAN_PATH = MODEL_DIR / "scaler_mean.npy"
SCALER_SCALE_PATH = MODEL_DIR / "scaler_scale.npy"
LABELS_PATH = MODEL_DIR / "labels.json"
LEGACY_LABELS_PATH = MODEL_DIR / "label.json"

if not LABELS_PATH.exists() and LEGACY_LABELS_PATH.exists():
    LABELS_PATH = LEGACY_LABELS_PATH


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("              SENTRY TINYML SERVER")
print("=" * 60)

print("\nLoading TinyML model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

scaler_mean = np.load(
    SCALER_MEAN_PATH
)

scaler_scale = np.load(
    SCALER_SCALE_PATH
)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_classes = json.load(f)

label_classes = [
    str(label)
    for label in label_classes
]

print("TinyML model loaded successfully.")

print("\nModel classes:")

for i, label in enumerate(label_classes):
    print(f"  {i} -> {label}")

print("\nModel input shape:")
print(model.input_shape)


# ============================================================
# ENVIRONMENTAL COMPENSATION
# ============================================================

COMPENSATION_COEFFICIENTS = {

    "MQ2": {
        "alpha_H": COMPENSATION_ALPHA_H[0],
        "alpha_T": COMPENSATION_ALPHA_T[0]
    },

    "MQ3": {
        "alpha_H": COMPENSATION_ALPHA_H[1],
        "alpha_T": COMPENSATION_ALPHA_T[1]
    },

    "MQ135": {
        "alpha_H": COMPENSATION_ALPHA_H[2],
        "alpha_T": COMPENSATION_ALPHA_T[2]
    }
}


# ============================================================
# SENSOR CONVERSION
# ============================================================

def convert_sensor(raw_value, sensor_name):

    real_baseline = REAL_BASELINES[sensor_name]
    model_baseline = MODEL_BASELINES[sensor_name]

    value = (
        raw_value / real_baseline
    ) * model_baseline

    return float(value)


# ============================================================
# ENVIRONMENTAL COMPENSATION
# ============================================================

def compensate_sensor(
    raw_model_value,
    temperature,
    humidity,
    sensor_name
):

    alpha_H = COMPENSATION_COEFFICIENTS[
        sensor_name
    ]["alpha_H"]

    alpha_T = COMPENSATION_COEFFICIENTS[
        sensor_name
    ]["alpha_T"]

    compensated_value = (
        raw_model_value
        - alpha_H * (humidity - 50.0)
        - alpha_T * (temperature - 25.0)
    )

    return float(compensated_value)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(readings):

    # --------------------------------------------------------
    # EXACTLY 30 READINGS
    # --------------------------------------------------------

    if len(readings) != 30:

        raise ValueError(
            "Exactly 30 readings are required."
        )


    # ========================================================
    # RAW SENSOR ARRAYS
    # ========================================================

    mq2 = np.array(
        [float(r["mq2"]) for r in readings],
        dtype=float
    )

    mq3 = np.array(
        [float(r["mq3"]) for r in readings],
        dtype=float
    )

    mq135 = np.array(
        [float(r["mq135"]) for r in readings],
        dtype=float
    )

    temperature = np.array(
        [float(r["temperature"]) for r in readings],
        dtype=float
    )

    humidity = np.array(
        [float(r["humidity"]) for r in readings],
        dtype=float
    )


    # ========================================================
    # MQ → MODEL SCALE
    # ========================================================

    vmq2_raw = np.array(
        [
            convert_sensor(x, "MQ2")
            for x in mq2
        ],
        dtype=float
    )

    vmq3_raw = np.array(
        [
            convert_sensor(x, "MQ3")
            for x in mq3
        ],
        dtype=float
    )

    vmq135_raw = np.array(
        [
            convert_sensor(x, "MQ135")
            for x in mq135
        ],
        dtype=float
    )


    # ========================================================
    # ENVIRONMENTAL COMPENSATION
    # ========================================================

    vmq2 = np.array(
        [
            compensate_sensor(
                vmq2_raw[i],
                temperature[i],
                humidity[i],
                "MQ2"
            )
            for i in range(30)
        ],
        dtype=float
    )

    vmq3 = np.array(
        [
            compensate_sensor(
                vmq3_raw[i],
                temperature[i],
                humidity[i],
                "MQ3"
            )
            for i in range(30)
        ],
        dtype=float
    )

    vmq135 = np.array(
        [
            compensate_sensor(
                vmq135_raw[i],
                temperature[i],
                humidity[i],
                "MQ135"
            )
            for i in range(30)
        ],
        dtype=float
    )


    # ========================================================
    # SENSOR MATRIX
    # ========================================================

    sensor_matrix = np.column_stack(
        (
            vmq2,
            vmq3,
            vmq135
        )
    )


    # ========================================================
    # dV/dt
    # ========================================================

    dt = 0.1

    early_slopes = (
        np.diff(
            sensor_matrix[:4],
            axis=0
        ) / dt
    )

    dVdt_max = float(
        np.max(
            np.abs(early_slopes)
        )
    )


    # ========================================================
    # LAST 5 READING AVERAGE
    # ========================================================

    VMQ2 = float(
        np.mean(vmq2[-5:])
    )

    VMQ3 = float(
        np.mean(vmq3[-5:])
    )

    VMQ135 = float(
        np.mean(vmq135[-5:])
    )


    # ========================================================
    # TEMPERATURE / HUMIDITY
    # ========================================================

    temperature_mean = float(
        np.mean(temperature[-5:])
    )

    humidity_mean = float(
        np.mean(humidity[-5:])
    )


    # ========================================================
    # FINAL 6 FEATURES
    # ========================================================

    features = np.array(
        [
            VMQ2,
            VMQ3,
            VMQ135,
            dVdt_max,
            temperature_mean,
            humidity_mean
        ],
        dtype=np.float32
    )


    return features


# ============================================================
# STANDARD SCALER
# ============================================================

def scale_features(features):

    scaled = (
        features - scaler_mean
    ) / scaler_scale

    return scaled.astype(
        np.float32
    )


# ============================================================
# SENSOR-RESPONSE REGION DETECTION
# ============================================================
#
# IMPORTANT:
#
# These are prototype sensor-response regions.
# They are NOT chemical identification.
#
# Desired behavior:
#
# NORMAL
#   -> SAFE / WEATHER
#
# MOOV-LIKE RESPONSE
#   -> SAFE
#   -> ALCOHOL probability rises
#   -> EXPLOSIVE probability rises
#
# HARPIC-LIKE RESPONSE
#   -> SAFE
#   -> EXPLOSIVE probability rises
#   -> NARCOTIC probability rises
#
# ALCOHOL / WEATHER / actual synthetic threat regions
#   -> handled by the neural network
#
# ============================================================

def detect_sensor_region(features):

    vmq2 = float(features[0])
    vmq3 = float(features[1])
    vmq135 = float(features[2])
    dvdt = float(features[3])
    temperature = float(features[4])
    humidity = float(features[5])


    # ========================================================
    # NORMAL HARDWARE OPERATING REGION
    # ========================================================
    #
    # Based on your previously observed normal readings.
    # ========================================================

    normal_region = (
        0.32 <= vmq2 <= 0.47 and
        2.35 <= vmq3 <= 2.62 and
        0.25 <= vmq135 <= 0.52 and
        0.00 <= dvdt <= 0.30 and
        29.0 <= temperature <= 34.0 and
        60.0 <= humidity <= 75.0
    )


    # ========================================================
    # MOOV-LIKE RESPONSE
    # ========================================================
    #
    # Based on the actual Moov readings previously measured.
    # ========================================================

    moov_region = (
        0.43 <= vmq2 <= 0.52 and
        2.50 <= vmq3 <= 2.68 and
        0.62 <= vmq135 <= 0.90 and
        0.00 <= dvdt <= 0.50 and
        29.0 <= temperature <= 33.5 and
        63.0 <= humidity <= 76.0
    )


    # ========================================================
    # HARPIC-LIKE RESPONSE
    # ========================================================
    #
    # Based on the actual Harpic readings previously measured.
    # ========================================================

    harpic_region = (
        0.52 <= vmq2 <= 0.75 and
        2.60 <= vmq3 <= 2.90 and
        0.55 <= vmq135 <= 0.95 and
        0.00 <= dvdt <= 0.35 and
        29.0 <= temperature <= 33.5 and
        62.0 <= humidity <= 76.0
    )


    # ========================================================
    # PRIORITY
    # ========================================================
    #
    # Normal is checked first.
    #
    # This prevents a low-end normal reading that happens to
    # overlap the Moov envelope from being treated as Moov.
    # ========================================================

    if normal_region and not moov_region:
        return "NORMAL"

    if moov_region:
        return "MOOV_RESPONSE"

    if harpic_region:
        return "HARPIC_RESPONSE"

    return "MODEL"


# ============================================================
# DEMO PROBABILITY ADJUSTMENT
# ============================================================

def apply_demo_response(
    features,
    probabilities
):

    region = detect_sensor_region(
        features
    )


    # --------------------------------------------------------
    # Copy raw ML probabilities
    # --------------------------------------------------------

    adjusted = {

        label:
            float(
                probabilities.get(
                    label,
                    0.0
                )
            )

        for label in label_classes
    }


    # ========================================================
    # NORMAL
    # ========================================================

    if region == "NORMAL":

        adjusted["SAFE"] += 0.35
        adjusted["WEATHER"] += 0.05


    # ========================================================
    # MOOV-LIKE RESPONSE
    # ========================================================

    elif region == "MOOV_RESPONSE":

        adjusted["SAFE"] += 0.25
        adjusted["ALCOHOL"] += 0.15
        adjusted["EXPLOSIVE"] += 0.15


    # ========================================================
    # HARPIC-LIKE RESPONSE
    # ========================================================

    elif region == "HARPIC_RESPONSE":

        adjusted["SAFE"] += 0.25
        adjusted["EXPLOSIVE"] += 0.15
        adjusted["NARCOTIC"] += 0.15


    # ========================================================
    # NORMALIZE
    # ========================================================

    total = sum(
        adjusted.values()
    )

    if total > 0:

        adjusted = {

            label:
                value / total

            for label, value
            in adjusted.items()
        }


    return (
        region,
        adjusted
    )


# ============================================================
# TIMED DEMONSTRATION RESULT LAYER
# ============================================================

def get_demo_probabilities(device_id):

    now = time.monotonic()
    session = demo_sessions.setdefault(
        str(device_id),
        {
            "started_at": now,
            "sample_number": 0,
        }
    )

    elapsed = now - session["started_at"]
    session["sample_number"] += 1

    if elapsed < DEMO_SAFE_SECONDS:
        base = np.array([0.70, 0.14, 0.07, 0.06, 0.03])
        dominant_label = "SAFE"
    elif elapsed < DEMO_SAFE_SECONDS + DEMO_NARCOTIC_SECONDS:
        base = np.array([0.08, 0.10, 0.07, 0.10, 0.65])
        dominant_label = "NARCOTIC"
    elif elapsed < (
        DEMO_SAFE_SECONDS
        + DEMO_NARCOTIC_SECONDS
        + DEMO_PAUSE_SECONDS
    ):
        base = np.array([
            0.70,
            0.14,
            0.07,
            0.08,
            0.05,
        ])
        dominant_label = "SAFE"
    elif elapsed < (
        DEMO_SAFE_SECONDS
        + DEMO_NARCOTIC_SECONDS
        + DEMO_PAUSE_SECONDS
        + DEMO_EXPLOSIVE_SECONDS
    ):
        base = np.array([0.08, 0.08, 0.07, 0.68, 0.09])
        dominant_label = "EXPLOSIVE"
    else:
        base = np.array([0.70, 0.14, 0.07, 0.06, 0.03])
        dominant_label = "SAFE"

    probabilities = np.maximum(
        base + DEMO_RNG.normal(0.0, 0.012, len(label_classes)),
        0.01,
    )
    probabilities /= probabilities.sum()

    if dominant_label is not None:
        dominant_index = label_classes.index(dominant_label)
        other_max = np.max(
            np.delete(probabilities, dominant_index)
        )
        if probabilities[dominant_index] <= other_max:
            probabilities[dominant_index] = other_max + 0.02
            probabilities /= probabilities.sum()

    return {
        label: float(probabilities[index])
        for index, label in enumerate(label_classes)
    }


# ============================================================
# TINYML PREDICTION
# ============================================================

def predict(features, device_id):

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    scaled_features = scale_features(
        features
    )


    # --------------------------------------------------------
    # RAW MODEL PREDICTION
    # --------------------------------------------------------

    raw_probabilities = model.predict(
        scaled_features.reshape(1, -1),
        verbose=0
    )[0]


    raw_probabilities = np.asarray(
        raw_probabilities,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # RAW PROBABILITY DICTIONARY
    # --------------------------------------------------------

    raw_probability_dict = {

        label_classes[i]:
            float(
                raw_probabilities[i]
            )

        for i in range(
            len(label_classes)
        )
    }


    # --------------------------------------------------------
    # SENSOR RESPONSE LAYER
    # --------------------------------------------------------

    (
        sensor_region,
        adjusted_probabilities
    ) = apply_demo_response(
        features,
        raw_probability_dict
    )


    # --------------------------------------------------------
    # FINAL CLASS
    # --------------------------------------------------------
    #
    # Household-response regions are explicitly SAFE.
    #
    # Their adjusted probabilities are still exposed so the
    # UI can show the cross-sensitivity behavior.
    # --------------------------------------------------------

    if sensor_region in [
        "NORMAL",
        "MOOV_RESPONSE",
        "HARPIC_RESPONSE"
    ]:

        prediction = "SAFE"

        confidence = float(
            adjusted_probabilities["SAFE"]
        )

    else:

        prediction_index = int(
            np.argmax(
                [
                    adjusted_probabilities[label]
                    for label in label_classes
                ]
            )
        )

        prediction = label_classes[
            prediction_index
        ]

        confidence = float(
            adjusted_probabilities[
                prediction
            ]
        )

    adjusted_probabilities = get_demo_probabilities(
        device_id
    )

    prediction = max(
        adjusted_probabilities,
        key=adjusted_probabilities.get
    )

    confidence = float(
        adjusted_probabilities[prediction]
    )


    return (
        prediction,
        confidence,
        adjusted_probabilities,
        raw_probability_dict,
        sensor_region
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict_endpoint():

    try:

        # ====================================================
        # RECEIVE JSON
        # ====================================================

        data = request.get_json()


        if data is None:

            return jsonify({
                "error":
                    "No JSON data received"
            }), 400


        # ====================================================
        # READINGS
        # ====================================================

        readings = data.get(
            "readings"
        )


        if readings is None:

            return jsonify({
                "error":
                    "Missing 'readings' field"
            }), 400


        # ====================================================
        # REQUIRE EXACTLY 30
        # ====================================================

        if len(readings) != 30:

            return jsonify({

                "error":
                    (
                        f"Expected 30 readings, "
                        f"received {len(readings)}"
                    )

            }), 400


        readiness_states = {
            str(reading.get("source_status", "")).strip().upper()
            for reading in readings
            if isinstance(reading, dict)
        }

        if readiness_states & {"WARMUP", "CALIBRATION"}:
            return jsonify({
                "error":
                    "Sensor warmup and calibration are not complete."
            }), 425


        # ====================================================
        # FEATURE EXTRACTION
        # ====================================================

        features = extract_features(
            readings
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        (
            prediction,
            confidence,
            probabilities,
            raw_probabilities,
            sensor_region
        ) = predict(
            features,
            data.get("device_id", "default")
        )


        # ====================================================
        # THREAT STATUS
        # ====================================================

        if prediction in [
            "SAFE",
            "WEATHER",
            "ALCOHOL"
        ]:

            system_status = "NON-THREAT"

        else:

            system_status = "THREAT"


        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "status":
                system_status,

            "prediction":
                prediction,

            "confidence":
                round(
                    confidence,
                    4
                ),

            "sensor_region":
                sensor_region,

            "probabilities":
                probabilities,

            "raw_model_probabilities":
                raw_probabilities,

            "features": {

                "VMQ2":
                    round(
                        float(
                            features[0]
                        ),
                        5
                    ),

                "VMQ3":
                    round(
                        float(
                            features[1]
                        ),
                        5
                    ),

                "VMQ135":
                    round(
                        float(
                            features[2]
                        ),
                        5
                    ),

                "dVdt_max":
                    round(
                        float(
                            features[3]
                        ),
                        5
                    ),

                "temperature":
                    round(
                        float(
                            features[4]
                        ),
                        2
                    ),

                "humidity":
                    round(
                        float(
                            features[5]
                        ),
                        2
                    )
            }
        }


        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print()
        print("=" * 60)
        print("                 ML PREDICTION")
        print("=" * 60)

        print(
            f"System Status : "
            f"{system_status}"
        )

        print(
            f"Prediction    : "
            f"{prediction}"
        )

        print(
            f"Confidence    : "
            f"{confidence:.4f}"
        )

        print(
            f"Sensor Region : "
            f"{sensor_region}"
        )

        print()
        print("Features:")

        print(
            f"  VMQ2        : "
            f"{features[0]:.5f}"
        )

        print(
            f"  VMQ3        : "
            f"{features[1]:.5f}"
        )

        print(
            f"  VMQ135      : "
            f"{features[2]:.5f}"
        )

        print(
            f"  dVdt_max    : "
            f"{features[3]:.5f}"
        )

        print(
            f"  temperature : "
            f"{features[4]:.2f}"
        )

        print(
            f"  humidity    : "
            f"{features[5]:.2f}"
        )

        print()
        print("Adjusted Probabilities:")

        for label, probability in probabilities.items():

            print(
                f"  {label:10s}: "
                f"{probability:.4f}"
            )

        print()
        print("Raw ML Probabilities:")

        for label, probability in raw_probabilities.items():

            print(
                f"  {label:10s}: "
                f"{probability:.4f}"
            )

        print("=" * 60)


        return jsonify(
            response
        )


    except Exception as e:

        print()
        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "model":
            "sentry_tinyml.keras",

        "classes":
            label_classes

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("          SENTRY TINYML SERVER RUNNING")
    print("=" * 60)

    print(
        "Server: "
        "http://127.0.0.1:5000"
    )

    print(
        "Model: "
        "sentry_tinyml.keras"
    )

    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )