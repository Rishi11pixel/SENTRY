from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
import tensorflow as tf
import sys


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
# MODEL PATHS
# ============================================================

MODEL_DIR = ROOT / "ml" / "models" / "tinyml"

MODEL_PATH = MODEL_DIR / "sentry_tinyml.keras"
SCALER_MEAN_PATH = MODEL_DIR / "scaler_mean.npy"
SCALER_SCALE_PATH = MODEL_DIR / "scaler_scale.npy"
LABELS_PATH = MODEL_DIR / "label_classes.npy"


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

label_classes = np.load(
    LABELS_PATH,
    allow_pickle=True
)

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
    #
    # ONLY THE THREE MQ SENSORS ARE USED.
    #
    # SEN0567 IS COMPLETELY REMOVED.
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
    #
    # Calculated from the first 4 readings of the
    # three MQ sensors, exactly as defined during training.
    #
    # Sampling interval = 100 ms = 0.1 s
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
    #
    # MUST MATCH TRAINING ORDER:
    #
    # VMQ2
    # VMQ3
    # VMQ135
    # dVdt_max
    # temperature
    # humidity
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
# TINYML PREDICTION
# ============================================================

def predict(features):

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    scaled_features = scale_features(
        features
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict(
        scaled_features.reshape(1, -1),
        verbose=0
    )[0]


    probabilities = np.asarray(
        probabilities,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # ARGMAX
    # --------------------------------------------------------

    prediction_index = int(
        np.argmax(probabilities)
    )

    prediction = label_classes[
        prediction_index
    ]


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = float(
        probabilities[prediction_index]
    )


    # --------------------------------------------------------
    # PROBABILITY DICTIONARY
    # --------------------------------------------------------

    probability_dict = {

        label_classes[i]:
            float(probabilities[i])

        for i in range(
            len(label_classes)
        )
    }


    return (
        prediction,
        confidence,
        probability_dict
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
        # REQUIRE 30
        # ====================================================

        if len(readings) != 30:

            return jsonify({

                "error":
                (
                    f"Expected 30 readings, "
                    f"received {len(readings)}"
                )

            }), 400


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
            probabilities
        ) = predict(features)


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

            "probabilities":
                probabilities,

            "features": {

                "VMQ2":
                    round(
                        float(features[0]),
                        5
                    ),

                "VMQ3":
                    round(
                        float(features[1]),
                        5
                    ),

                "VMQ135":
                    round(
                        float(features[2]),
                        5
                    ),

                "dVdt_max":
                    round(
                        float(features[3]),
                        5
                    ),

                "temperature":
                    round(
                        float(features[4]),
                        2
                    ),

                "humidity":
                    round(
                        float(features[5]),
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
            f"System Status : {system_status}"
        )

        print(
            f"Prediction    : {prediction}"
        )

        print(
            f"Confidence    : {confidence:.4f}"
        )

        print()
        print("Features:")

        print(
            f"  VMQ2        : {features[0]:.5f}"
        )

        print(
            f"  VMQ3        : {features[1]:.5f}"
        )

        print(
            f"  VMQ135      : {features[2]:.5f}"
        )

        print(
            f"  dVdt_max    : {features[3]:.5f}"
        )

        print(
            f"  temperature : {features[4]:.2f}"
        )

        print(
            f"  humidity    : {features[5]:.2f}"
        )

        print()
        print("Probabilities:")

        for label, probability in probabilities.items():

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