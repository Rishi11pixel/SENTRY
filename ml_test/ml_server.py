from flask import Flask, request, jsonify
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.config import MODEL_BASELINES, REAL_BASELINES


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = ROOT / "models" / "random_forest"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
LABELS_PATH = MODEL_DIR / "label_encoder.joblib"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print()
print("=" * 50)
print("          SENTRY ML SERVER")
print("=" * 50)

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABELS_PATH)

print("Model loaded successfully.")

print("\nML Classes:")

for i, label in enumerate(label_encoder):
    print(f"  {i} -> {label}")


# ============================================================
# ENVIRONMENTAL COMPENSATION
# ============================================================

COMPENSATION_COEFFICIENTS = {

    "MQ2": {
        "alpha_H": 0.0018,
        "alpha_T": 0.0008
    },

    "MQ3": {
        "alpha_H": 0.0014,
        "alpha_T": 0.0007
    },

    "MQ135": {
        "alpha_H": 0.0022,
        "alpha_T": 0.0009
    },

    "SEN0567": {
        "alpha_H": 0.0016,
        "alpha_T": 0.0008
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
# SEN0567
# ============================================================

def convert_sen0567(value):

    # ESP32 already sends SEN0567 as volts.
    return float(value)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(readings):

    # --------------------------------------------------------
    # REQUIRE EXACTLY 30 READINGS
    # --------------------------------------------------------

    if len(readings) != 30:

        raise ValueError(
            "Exactly 30 readings are required."
        )


    # ========================================================
    # RAW ARRAYS
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

    sen0567 = np.array(
        [float(r["sen0567"]) for r in readings],
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
    # CONVERT MQ SENSORS TO MODEL SCALE
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
    # SEN0567
    # ========================================================

    vsen0567_raw = np.array(
        [
            convert_sen0567(x)
            for x in sen0567
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

    vsen0567 = np.array(
        [
            compensate_sensor(
                vsen0567_raw[i],
                temperature[i],
                humidity[i],
                "SEN0567"
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
            vmq135,
            vsen0567
        )
    )


    # ========================================================
    # dV/dt
    # ========================================================
    #
    # ONLY FIRST FOUR READINGS
    #
    # 0.0 -> 0.1
    # 0.1 -> 0.2
    # 0.2 -> 0.3
    #
    # Same feature definition used during training.
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
    # FINAL 5-READING AVERAGE
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

    VSEN0567 = float(
        np.mean(vsen0567[-5:])
    )


    # ========================================================
    # FINAL TEMPERATURE / HUMIDITY
    # ========================================================

    temperature_mean = float(
        np.mean(temperature[-5:])
    )

    humidity_mean = float(
        np.mean(humidity[-5:])
    )


    # ========================================================
    # FINAL 7 FEATURES
    # ========================================================

    features = np.array(
        [
            VMQ2,
            VMQ3,
            VMQ135,
            VSEN0567,
            dVdt_max,
            temperature_mean,
            humidity_mean
        ],
        dtype=np.float32
    )


    return features


# ============================================================
# RANDOM FOREST PREDICTION
# ============================================================

def predict(features):

    # ========================================================
    # SCALE USING TRAINING SCALER
    # ========================================================

    scaled_features = scaler.transform(
        features.reshape(1, -1)
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction_encoded = model.predict(
        scaled_features
    )[0]


    prediction = str(
        prediction_encoded
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    probabilities_array = model.predict_proba(
        scaled_features
    )[0]


    # ========================================================
    # MAP PROBABILITIES TO CLASS NAMES
    # ========================================================

    probability_dict = {

        str(class_name):
            float(probabilities_array[i])

        for i, class_name in enumerate(
            model.classes_
        )
    }


    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = float(
        np.max(probabilities_array)
    )


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
        # GET READINGS
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
        # REQUIRE 30 READINGS
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
        # ML PREDICTION
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

                "VSEN0567":
                    round(
                        float(features[3]),
                        5
                    ),

                "dVdt_max":
                    round(
                        float(features[4]),
                        5
                    ),

                "temperature":
                    round(
                        float(features[5]),
                        2
                    ),

                "humidity":
                    round(
                        float(features[6]),
                        2
                    )
            }
        }


        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print()
        print("=" * 50)
        print("             ML PREDICTION")
        print("=" * 50)

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
            f"  VSEN0567    : {features[3]:.5f}"
        )

        print(
            f"  dVdt_max    : {features[4]:.5f}"
        )

        print(
            f"  temperature : {features[5]:.2f}"
        )

        print(
            f"  humidity    : {features[6]:.2f}"
        )

        print()
        print("Probabilities:")

        for label, probability in probabilities.items():

            print(
                f"  {label:10s}: "
                f"{probability:.4f}"
            )

        print("=" * 50)


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
            "random_forest.joblib"

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("       SENTRY ML SERVER RUNNING")
    print("=" * 50)

    print(
        "Server: "
        "http://127.0.0.1:5000"
    )

    print(
        "Model: "
        "random_forest.joblib"
    )

    print("=" * 50)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )