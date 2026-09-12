from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
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

MODEL_PATH = Path(
    r"C:\SENTRY\models\tinyml\sentry_tinyml.keras"
)

SCALER_MEAN_PATH = Path(
    r"C:\SENTRY\models\tinyml\scaler_mean.npy"
)

SCALER_SCALE_PATH = Path(
    r"C:\SENTRY\models\tinyml\scaler_scale.npy"
)

LABELS_PATH = Path(
    r"C:\SENTRY\models\tinyml\label_classes.npy"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print()
print("=" * 40)
print("       SENTRY ML SERVER")
print("=" * 40)

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

scaler_mean = np.load(
    SCALER_MEAN_PATH
)

scaler_scale = np.load(
    SCALER_SCALE_PATH
)


# ============================================================
# LOAD LABEL CLASSES
# ============================================================

label_classes = np.load(
    LABELS_PATH,
    allow_pickle=True
)

print("\nML Classes:")

for i, label in enumerate(label_classes):
    print(f"  {i} -> {label}")


# ============================================================
# MODEL TRAINING BASELINES
# ============================================================
#
# These are the baseline values in the model's sensor scale.
#
# IMPORTANT:
# SEN0567 is assumed to already arrive in the same model-space
# voltage/unit used during training.
#
# If SEN0567 arrives as an ADC count from ESP32 instead,
# this section MUST be changed after confirming its calibration.
# ============================================================

# ============================================================
# REAL ESP32 SENSOR BASELINES
# ============================================================
#
# These are raw ADC baselines from the ESP32.
# ============================================================

# ============================================================
# ENVIRONMENTAL COMPENSATION COEFFICIENTS
# ============================================================
#
# V_comp = V_raw
#          - alpha_H * (Humidity - 50)
#          - alpha_T * (Temperature - 25)
#
# Compensation is performed AFTER conversion to model scale
# and BEFORE dV/dt calculation.
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

    alpha_H = (
        COMPENSATION_COEFFICIENTS[
            sensor_name
        ]["alpha_H"]
    )

    alpha_T = (
        COMPENSATION_COEFFICIENTS[
            sensor_name
        ]["alpha_T"]
    )

    compensated_value = (
        raw_model_value
        - alpha_H * (humidity - 50.0)
        - alpha_T * (temperature - 25.0)
    )

    return float(compensated_value)


# ============================================================
# SEN0567 CONVERSION
# ============================================================
#
# IMPORTANT ASSUMPTION:
#
# The ESP32 sends SEN0567 in the same voltage/model-space
# expected by the trained model.
#
# The ESP32 value is forwarded in the model-space units used by
# the training generator. If a future device sends ADC counts,
# its calibration must be established before changing this function.
# ============================================================

def convert_sen0567(value):

    return float(value)


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
        [
            float(r["mq2"])
            for r in readings
        ],
        dtype=float
    )


    mq3 = np.array(
        [
            float(r["mq3"])
            for r in readings
        ],
        dtype=float
    )


    mq135 = np.array(
        [
            float(r["mq135"])
            for r in readings
        ],
        dtype=float
    )


    sen0567 = np.array(
        [
            float(r["sen0567"])
            for r in readings
        ],
        dtype=float
    )


    temperature = np.array(
        [
            float(r["temperature"])
            for r in readings
        ],
        dtype=float
    )


    humidity = np.array(
        [
            float(r["humidity"])
            for r in readings
        ],
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
    #
    # IMPORTANT:
    #
    # Compensation happens BEFORE dV/dt.
    #
    # Each sensor is compensated using the temperature and
    # humidity corresponding to the same reading.
    # ========================================================

    vmq2 = np.array(
        [
            compensate_sensor(
                vmq2_raw[i],
                temperature[i],
                humidity[i],
                "MQ2"
            )
            for i in range(len(readings))
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
            for i in range(len(readings))
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
            for i in range(len(readings))
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
            for i in range(len(readings))
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
    # SENSOR SAMPLING:
    #
    # Reading 1 -> 0.0 s
    # Reading 2 -> 0.1 s
    # Reading 3 -> 0.2 s
    # Reading 4 -> 0.3 s
    #
    # The trained pipeline uses ONLY THESE FIRST FOUR
    # readings to calculate the maximum absolute slope.
    #
    # Therefore:
    #
    # np.diff(sensor_matrix[:4])
    #
    # produces 3 slope intervals:
    #
    # 0.0 -> 0.1 s
    # 0.1 -> 0.2 s
    # 0.2 -> 0.3 s
    #
    # DO NOT calculate dV/dt over all 30 readings.
    # ========================================================

    dt = 0.1


    early_slopes = (
        np.diff(
            sensor_matrix[:4],
            axis=0
        ) / dt
    )


    # Maximum absolute slope across:
    #
    # - MQ2
    # - MQ3
    # - MQ135
    # - SEN0567
    #
    # and across the three initial time intervals.

    dVdt_max = float(
        np.max(
            np.abs(early_slopes)
        )
    )


    # ========================================================
    # FINAL SENSOR VALUES
    # ========================================================
    #
    # Use the LAST 5 readings.
    #
    # 5 readings × 0.1 s = 0.5-second final averaging window.
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
        temperature[-1]
    )


    humidity_mean = float(
        humidity[-1]
    )


    # ========================================================
    # FINAL 7-FEATURE VECTOR
    # ========================================================
    #
    # MUST remain in exactly this order:
    #
    # [VMQ2,
    #  VMQ3,
    #  VMQ135,
    #  VSEN0567,
    #  dVdt_max,
    #  temperature,
    #  humidity]
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
        dtype=float
    )


    return features


# ============================================================
# ML PREDICTION
# ============================================================

def predict(features):

    # --------------------------------------------------------
    # STANDARD SCALING
    # --------------------------------------------------------

    scaled_features = (
        features - scaler_mean
    ) / scaler_scale


    scaled_features = (
        scaled_features.reshape(1, -1)
    )


    # --------------------------------------------------------
    # TRAINED KERAS MODEL
    # --------------------------------------------------------

    probabilities = model.predict(
        scaled_features,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # HIGHEST PROBABILITY CLASS
    # --------------------------------------------------------

    prediction_index = int(
        np.argmax(probabilities)
    )


    prediction = str(
        label_classes[prediction_index]
    )


    # --------------------------------------------------------
    # MODEL CONFIDENCE
    #
    # Directly from model output.
    # --------------------------------------------------------

    confidence = float(
        probabilities[prediction_index]
    )


    # --------------------------------------------------------
    # ALL CLASS PROBABILITIES
    # --------------------------------------------------------

    probability_dict = {

        str(label_classes[i]):
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
        # APPLICATION STATUS
        # ====================================================
        #
        # SAFE environmental classes:
        #
        #   SAFE
        #   WEATHER
        #
        # Everything else is treated as ALERT.
        #
        # This does NOT modify ML confidence or probabilities.
        # ====================================================

        if prediction in [
            "SAFE",
            "WEATHER",
            "AMBIENT_CLEAN",
            "WEATHER_DRIFT"
        ]:

            system_status = "SAFE"

        else:

            system_status = "ALERT"


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
        print("=" * 40)
        print("           ML PREDICTION")
        print("=" * 40)

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
            f"  VSEN0567    : "
            f"{features[3]:.5f}"
        )

        print(
            f"  dVdt_max    : "
            f"{features[4]:.5f}"
        )

        print(
            f"  temperature : "
            f"{features[5]:.2f}"
        )

        print(
            f"  humidity    : "
            f"{features[6]:.2f}"
        )

        print("=" * 40)


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
            "sentry_tinyml.keras"

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 40)
    print("       SENTRY ML SERVER RUNNING")
    print("=" * 40)

    print(
        "Server: "
        "http://127.0.0.1:5000"
    )

    print("=" * 40)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )