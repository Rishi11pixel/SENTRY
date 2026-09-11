import requests
import re
import time
import os
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SERVER_URL = os.getenv("SENTRY_BACKEND_URL", "http://127.0.0.1:8000/api/v1/devices/SENTRY-032/readings")
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_READINGS_FILE = REPO_ROOT / "sentry_fake_readings_1800.txt"
READINGS_FILE = DEFAULT_READINGS_FILE
WAIT_TIME = 0.1

# ============================================================
# LOAD ALL READINGS FROM TXT
# ============================================================

def load_readings():

    print("\nReading:", READINGS_FILE)

    if not READINGS_FILE.is_file():
        raise FileNotFoundError(
            f"Canonical fake readings file not found: {READINGS_FILE}"
        )

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
        r"MQ-135\s*\(Air Quality\):\s*(\d+)\s*"
        r"Fermion NH3\s*\(SEN0567\):\s*([-+]?\d+(?:\.\d+)?)\s*"
        r"(?:STATUS:\s*([^\r\n]+))?",
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
        sen0567 = float(match[5])
        source_status = match[6].strip() if match[6] else None


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

            "sen0567": sen0567
        })

        if source_status:
            readings[-1]["source_status"] = source_status


    return readings


def send_reading(reading, reading_number):
    payload = {**reading, "battery": 84, "signal": 92}
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"Reading {reading_number}: HTTP {response.status_code} - {response.text}")
            return False

        result = response.json()
        print(
            f"{reading_number:03d} | buffer={result['buffer_count']}/{result['window_size']}"
        )
        if result.get("prediction"):
            prediction = result["prediction"]
            print(
                f"  Prediction: {prediction.get('prediction')} "
                f"({float(prediction.get('confidence', 0)) * 100:.2f}%)"
            )
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to SENTRY backend. Run: python backend/app.py")
        return False
    except (KeyError, ValueError, requests.RequestException) as exc:
        print(f"ERROR sending reading {reading_number}: {exc}")
        return False


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


    print(
        "\nStarting transmission..."
    )

    for reading_number, reading in enumerate(readings, start=1):
        if not send_reading(reading, reading_number):
            print("\nStopping.")
            break
        if reading_number < len(readings):
            time.sleep(WAIT_TIME)


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