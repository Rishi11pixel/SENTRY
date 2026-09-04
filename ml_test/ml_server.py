from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from pathlib import Path


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

MODEL_BASELINES = {
    "MQ2": 0.82,
    "MQ3": 0.74,
    "MQ135": 0.91,
    "SEN0568": 0.68
}


# ============================================================
# REAL ESP32 SENSOR BASELINES
# ============================================================

REAL_BASELINES = {
    "MQ2": 3069.0,
    "MQ3": 530.33,
    "MQ135": 1043.27
}


# ============================================================
# SEN0568
# ============================================================

# SEN0568 sensor is not currently available
# so we use the same value used for the model.

VSEN0568_VALUE = 0.68


# ============================================================
# SENSOR CONVERSION
# ============================================================

def convert_sensor(raw_value, sensor_name):

    real_baseline = REAL_BASELINES[sensor_name]

    model_baseline = MODEL_BASELINES[sensor_name]

    value = (
        raw_value / real_baseline
    ) * model_baseline

    value = np.clip(
        value,
        0.02,
        3.25
    )

    return float(value)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(readings):

    if len(readings) != 30:

        raise ValueError(
            "Exactly 30 readings are required."
        )


    # --------------------------------------------------------
    # RAW SENSOR ARRAYS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CONVERT SENSOR VALUES TO MODEL SCALE
    # --------------------------------------------------------

    vmq2 = np.array(
        [
            convert_sensor(x, "MQ2")
            for x in mq2
        ]
    )

    vmq3 = np.array(
        [
            convert_sensor(x, "MQ3")
            for x in mq3
        ]
    )

    vmq135 = np.array(
        [
            convert_sensor(x, "MQ135")
            for x in mq135
        ]
    )


    # --------------------------------------------------------
    # SEN0568
    # --------------------------------------------------------

    vsen0568 = np.full(
        len(readings),
        VSEN0568_VALUE
    )


    # --------------------------------------------------------
    # SENSOR MATRIX
    # --------------------------------------------------------

    sensor_matrix = np.column_stack(
        (
            vmq2,
            vmq3,
            vmq135,
            vsen0568
        )
    )


    # --------------------------------------------------------
    # dV/dt
    #
    # Sensor readings arrive every 0.1 seconds.
    # --------------------------------------------------------

    dt = 0.1


    # Use first four readings exactly as in
    # the existing implementation.

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


    # --------------------------------------------------------
    # FINAL SENSOR VALUES
    #
    # Average of the last 5 readings.
    # --------------------------------------------------------

    VMQ2 = float(
        np.mean(vmq2[-5:])
    )

    VMQ3 = float(
        np.mean(vmq3[-5:])
    )

    VMQ135 = float(
        np.mean(vmq135[-5:])
    )

    VSEN0568 = float(
        np.mean(vsen0568[-5:])
    )


    # --------------------------------------------------------
    # TEMPERATURE / HUMIDITY
    # --------------------------------------------------------

    temperature_mean = float(
        np.mean(temperature[-5:])
    )

    humidity_mean = float(
        np.mean(humidity[-5:])
    )


    # --------------------------------------------------------
    # FEATURE VECTOR
    #
    # IMPORTANT:
    # Keep exactly the same feature order used
    # during ML model training.
    # --------------------------------------------------------

    features = np.array(
        [
            VMQ2,
            VMQ3,
            VMQ135,
            VSEN0568,
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
    # STANDARD SCALER
    #
    # This is only preprocessing.
    # It does NOT modify the model's output confidence.
    # --------------------------------------------------------

    scaled_features = (
        features - scaler_mean
    ) / scaler_scale


    scaled_features = scaled_features.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # ACTUAL TRAINED ML MODEL
    # --------------------------------------------------------

    probabilities = model.predict(
        scaled_features,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # SELECT HIGHEST-PROBABILITY CLASS
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
    # THIS VALUE COMES DIRECTLY FROM THE ML MODEL.
    #
    # DO NOT MODIFY IT.
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

        # ----------------------------------------------------
        # RECEIVE JSON
        # ----------------------------------------------------

        data = request.get_json()


        if data is None:

            return jsonify({
                "error":
                "No JSON data received"
            }), 400


        # ----------------------------------------------------
        # GET SENSOR READINGS
        # ----------------------------------------------------

        readings = data.get(
            "readings"
        )


        if readings is None:

            return jsonify({
                "error":
                "Missing 'readings' field"
            }), 400


        # ----------------------------------------------------
        # EXACTLY 30 READINGS REQUIRED
        # ----------------------------------------------------

        if len(readings) != 30:

            return jsonify({
                "error":
                (
                    f"Expected 30 readings, "
                    f"received {len(readings)}"
                )
            }), 400


        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        features = extract_features(
            readings
        )


        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        (
            prediction,
            confidence,
            probabilities
        ) = predict(features)


        # ----------------------------------------------------
        # SYSTEM STATUS
        #
        # IMPORTANT:
        #
        # This does NOT change:
        #   - ML prediction
        #   - ML confidence
        #   - ML probabilities
        #
        # It ONLY decides whether the application
        # displays SAFE or ALERT.
        # ----------------------------------------------------

        if prediction in [
            "AMBIENT_CLEAN",
            "WEATHER_DRIFT",
            "WEATHER"
        ]:

            system_status = "SAFE"

        else:

            system_status = "ALERT"


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

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

                "VSEN0568":
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


        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print()
        print("=" * 40)
        print("           ML PREDICTION")
        print("=" * 40)

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
            f"  VSEN0568    : "
            f"{features[3]:.2f}"
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


        return jsonify(response)


    except Exception as e:

        print()
        print(
            "Prediction error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
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
        "Server: http://127.0.0.1:5000"
    )

    print("=" * 40)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )