import requests
import re
import time


# ============================================================
# CONFIG
# ============================================================

SERVER_URL = "http://127.0.0.1:5000/predict"
READINGS_FILE = "sensor_readings.txt"
WAIT_TIME = 3

# Temporary assumption:
# SEN0567 is currently assumed to be in model-space units.
SEN0567_VALUE = 0.68


# ============================================================
# LOAD ALL READINGS FROM TXT
# ============================================================

def load_readings():

    print("\nReading:", READINGS_FILE)

    with open(
        READINGS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    # --------------------------------------------------------
    # Match every sensor-reading block in the TXT file.
    # --------------------------------------------------------

    pattern = re.compile(
        r"Temperature:\s*([-+]?\d+(?:\.\d+)?)\s*C\s*"
        r"Humidity:\s*([-+]?\d+(?:\.\d+)?)\s*%\s*"
        r"MQ-2\s*\(Gas\):\s*(\d+)\s*"
        r"MQ-3\s*\(Flying-Fish\):\s*(\d+)\s*"
        r"MQ-135\s*\(Air Quality\):\s*(\d+)",
        re.MULTILINE
    )


    matches = pattern.findall(text)


    readings = []


    for i, match in enumerate(matches):

        temperature = float(match[0])
        humidity = float(match[1])

        mq2 = float(match[2])
        mq3 = float(match[3])
        mq135 = float(match[4])


        readings.append({

            "timestamp": round(
                i * 0.1,
                1
            ),

            "temperature": temperature,

            "humidity": humidity,

            "mq2": mq2,

            "mq3": mq3,

            "mq135": mq135,

            # ------------------------------------------------
            # TEMPORARY SEN0567 VALUE
            # ------------------------------------------------
            "sen0567": SEN0567_VALUE
        })


    return readings


# ============================================================
# SEND ONE 3-SECOND WINDOW
# ============================================================

def send_window(
    window,
    window_number
):

    payload = {

        "device_id":
            "SENTRY-01",

        "readings":
            window
    }


    print(
        "\n========================================"
    )

    print(
        f"SENDING WINDOW #{window_number}"
    )

    print(
        "Readings:",
        len(window)
    )

    print(
        f"Time: "
        f"{window[0]['timestamp']:.1f}s"
        f" -> "
        f"{window[-1]['timestamp']:.1f}s"
    )

    print(
        "========================================"
    )


    # ========================================================
    # PRINT ACTUAL SENSOR VALUES
    # ========================================================

    print(
        "\nSensor data being sent:"
    )


    for i, reading in enumerate(window):

        print(

            f"{i+1:02d} | "

            f"T={reading['temperature']:.2f} C | "

            f"H={reading['humidity']:.2f}% | "

            f"MQ2={reading['mq2']:.0f} | "

            f"MQ3={reading['mq3']:.0f} | "

            f"MQ135={reading['mq135']:.0f} | "

            f"SEN0567={reading['sen0567']:.2f}"

        )


    # ========================================================
    # HTTP REQUEST
    # ========================================================

    try:

        response = requests.post(

            SERVER_URL,

            json=payload,

            timeout=10
        )


        print(
            "\nHTTP STATUS:"
        )

        print(
            response.status_code
        )


        # ====================================================
        # SHOW SERVER ERROR IF HTTP 500
        # ====================================================

        if response.status_code != 200:

            print(
                "\nSERVER ERROR RESPONSE:"
            )

            print(
                response.text
            )

            return False


        result = response.json()


        # ====================================================
        # MODEL RESPONSE
        # ====================================================

        print(
            "\n========== MODEL RESPONSE =========="
        )


        print(

            "Prediction:",

            result.get(
                "prediction"
            )

        )


        print(

            "Confidence:",

            result.get(
                "confidence"
            )

        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        print(
            "\nProbabilities:"
        )


        for (
            label,
            probability
        ) in result.get(
            "probabilities",
            {}
        ).items():

            print(

                f"  {label}: "
                f"{probability}"

            )


        # ====================================================
        # FEATURES
        # ====================================================

        print(
            "\nFeatures sent to model:"
        )


        for (
            name,
            value
        ) in result.get(
            "features",
            {}
        ).items():

            print(

                f"  {name}: "
                f"{value}"

            )


        print(
            "===================================="
        )


    except requests.exceptions.ConnectionError:

        print(
            "\nERROR: Cannot connect to ML server."
        )

        print(
            "Run: python ml_server.py"
        )

        return False


    except Exception as e:

        print(
            "\nERROR:",
            e
        )

        return False


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "       SENTRY SENSOR → ML"
    )

    print(
        "========================================"
    )


    # ========================================================
    # LOAD EVERYTHING FROM TXT
    # ========================================================

    readings = load_readings()


    print(

        f"\nTotal sensor readings found: "
        f"{len(readings)}"

    )


    if len(readings) < 30:

        print(

            "ERROR: TXT file contains fewer "
            "than 30 readings."

        )

        return


    # ========================================================
    # CREATE 30-SAMPLE WINDOWS
    # ========================================================

    windows = []


    for i in range(

        0,

        len(readings) - 29,

        30

    ):

        window = readings[
            i:i + 30
        ]


        windows.append(
            window
        )


    print(

        f"Complete 3-second windows: "
        f"{len(windows)}"

    )


    print(
        "\nStarting transmission..."
    )


    # ========================================================
    # SEND EVERY WINDOW
    # ========================================================

    for (

        window_number,

        window

    ) in enumerate(

        windows,

        start=1

    ):


        success = send_window(

            window,

            window_number

        )


        if not success:

            print(
                "\nStopping."
            )

            break


        # ====================================================
        # WAIT 3 SECONDS
        # ====================================================

        if (

            window_number
            < len(windows)

        ):

            print(

                "\nWaiting 3 seconds "
                "before next request..."

            )


            time.sleep(
                WAIT_TIME
            )


    print(
        "\n========================================"
    )

    print(
        "         TRANSMISSION COMPLETE"
    )

    print(
        "========================================"
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()