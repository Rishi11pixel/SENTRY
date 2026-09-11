from __future__ import annotations

import os
import threading
from collections import deque
from datetime import datetime, timezone
from math import isfinite
from typing import Any
from uuid import uuid4

import requests
from flask import Flask, jsonify, request


WINDOW_SIZE = 30
READING_FIELDS = ("temperature", "humidity", "mq2", "mq3", "mq135", "sen0567")
MODEL_LABELS = ("SAFE", "WEATHER", "ALCOHOL", "EXPLOSIVE", "NARCOTIC")
SAFE_LABELS = {"SAFE", "WEATHER"}
OPTIONAL_METADATA_FIELDS = ("source_status", "test_object")

BACKEND_HOST = os.getenv("SENTRY_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("SENTRY_BACKEND_PORT", "8000"))
ML_SERVER_URL = os.getenv("SENTRY_ML_URL", "http://127.0.0.1:5000")
ML_TIMEOUT_SECONDS = float(os.getenv("SENTRY_ML_TIMEOUT_SECONDS", "10"))
FRONTEND_ORIGINS = os.getenv(
    "SENTRY_FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8443,http://127.0.0.1:8443",
).split(",")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def display_result(prediction: str) -> str:
    return {
        "WEATHER": "CAUTION",
        "EXPLOSIVE": "EXPLOSIVE PROXY",
        "NARCOTIC": "NARCOTIC PROXY",
    }.get(prediction, prediction)


def default_device(device_id: str) -> dict[str, Any]:
    known = {
        "SENTRY-014": ("Platform 3", "Platform 3", 38, 32),
        "SENTRY-021": ("Platform 5", "Platform 5", 62, 52),
        "SENTRY-032": ("Entry Gate 2", "Platform 4", 18, 72),
        "SENTRY-008": ("Coaching Area", "Coaching Bay", 82, 58),
        "SENTRY-019": ("Entry Gate 1", "Main Entrance", 18, 38),
        "SENTRY-027": ("Platform 2", "Platform 2", 76, 78),
        "SENTRY-041": ("Waiting Hall", "Ground Floor", 62, 22),
    }
    location, platform, map_x, map_y = known.get(
        device_id, ("Unassigned", "Unassigned", 50, 50)
    )
    return {
        "id": device_id,
        "location": location,
        "platform": platform,
        "status": "OFFLINE",
        "battery": None,
        "signal": None,
        "health": None,
        "lastSync": None,
        "operatingTime": None,
        "lastResult": "SAFE",
        "confidence": 0,
        "mapX": map_x,
        "mapY": map_y,
        "latestReading": None,
        "latestPrediction": None,
        "registeredAt": utc_now(),
    }


app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in {item.strip() for item in FRONTEND_ORIGINS}:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

lock = threading.RLock()
devices: dict[str, dict[str, Any]] = {}
windows: dict[str, deque[dict[str, Any]]] = {}
predictions: dict[str, list[dict[str, Any]]] = {}
pending_windows: dict[str, deque[list[dict[str, Any]]]] = {}
incidents: dict[str, dict[str, Any]] = {}
logs: deque[dict[str, Any]] = deque(maxlen=500)

for _device_id in (
    "SENTRY-014",
    "SENTRY-021",
    "SENTRY-032",
    "SENTRY-008",
    "SENTRY-019",
    "SENTRY-027",
    "SENTRY-041",
):
    devices[_device_id] = default_device(_device_id)
    windows[_device_id] = deque(maxlen=WINDOW_SIZE)
    predictions[_device_id] = []
    pending_windows[_device_id] = deque()


def get_device(device_id: str) -> dict[str, Any]:
    if device_id not in devices:
        devices[device_id] = default_device(device_id)
        windows[device_id] = deque(maxlen=WINDOW_SIZE)
        predictions[device_id] = []
        pending_windows[device_id] = deque()
    return devices[device_id]


def add_log(device_id: str, event: str, level: str, location: str) -> None:
    logs.appendleft(
        {
            "timestamp": utc_now(),
            "device": device_id,
            "event": event,
            "location": location,
            "level": level,
        }
    )


def error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def validate_reading(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    missing = [field for field in READING_FIELDS if field not in payload]
    if missing:
        return None, f"Missing required sensor field(s): {', '.join(missing)}."

    reading = {}
    for field in READING_FIELDS:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"Sensor field '{field}' must be numeric."
        reading[field] = float(value)

    ranges = {
        "temperature": (-50, 100),
        "humidity": (0, 100),
        "mq2": (0, 100000),
        "mq3": (0, 100000),
        "mq135": (0, 100000),
        "sen0567": (0, 100),
    }
    for field, (minimum, maximum) in ranges.items():
        if not minimum <= reading[field] <= maximum:
            return None, f"Sensor field '{field}' is outside the allowed range."

    timestamp = payload.get("timestamp", utc_now())
    if isinstance(timestamp, bool):
        return None, "'timestamp' must be an ISO-8601 string or numeric elapsed time."
    if isinstance(timestamp, (int, float)):
        if not isfinite(float(timestamp)) or float(timestamp) < 0:
            return None, "Numeric 'timestamp' must be a finite, non-negative elapsed time."
        timestamp = float(timestamp)
    elif isinstance(timestamp, str):
        if not timestamp.strip():
            return None, "'timestamp' must be a non-empty ISO-8601 string when provided."
        try:
            datetime.fromisoformat(timestamp.strip().replace("Z", "+00:00"))
        except ValueError:
            return None, "String 'timestamp' must be ISO-8601 formatted."
    else:
        return None, "'timestamp' must be an ISO-8601 string or numeric elapsed time."
    reading["timestamp"] = timestamp

    for field in ("battery", "signal"):
        if field in payload:
            value = payload[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None, f"Optional field '{field}' must be numeric."
            if not 0 <= float(value) <= 100:
                return None, f"Optional field '{field}' must be between 0 and 100."
            reading[field] = float(value)

    for field in OPTIONAL_METADATA_FIELDS:
        if field in payload:
            value = payload[field]
            if value is not None and not isinstance(value, str):
                return None, f"Optional metadata field '{field}' must be a string when provided."
            if value is not None:
                reading[field] = value.strip()

    return reading, None


def validate_ml_response(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("ML server returned a non-object response.")
    prediction = result.get("prediction")
    if prediction not in MODEL_LABELS:
        raise ValueError("ML server returned an unknown prediction label.")

    confidence = result.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("ML server returned a non-numeric confidence.")
    if not isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise ValueError("ML server returned an invalid confidence range.")

    for field in ("probabilities", "features"):
        if field not in result:
            continue
        values = result[field]
        if not isinstance(values, dict):
            raise ValueError(f"ML server returned invalid '{field}'.")
        for value in values.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"ML server returned non-numeric '{field}'.")
        if field == "probabilities" and any(not 0 <= float(value) <= 1 for value in values.values()):
            raise ValueError("ML server returned an invalid probability range.")

    expected_features = {"VMQ2", "VMQ3", "VMQ135", "VSEN0567", "dVdt_max", "temperature", "humidity"}
    if "features" in result and set(result["features"]) != expected_features:
        raise ValueError("ML server returned an unexpected feature structure.")
    return result


def call_ml_server(device_id: str, window: list[dict[str, Any]]) -> dict[str, Any]:
    response = requests.post(
        f"{ML_SERVER_URL.rstrip('/')}/predict",
        json={"device_id": device_id, "readings": window},
        timeout=ML_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def create_prediction(device_id: str, window: list[dict[str, Any]]) -> dict[str, Any] | None:
    device = get_device(device_id)
    try:
        model_result = validate_ml_response(call_ml_server(device_id, window))
    except (requests.RequestException, ValueError) as exc:
        add_log(device_id, f"ML inference unavailable: {exc}", "WARNING", device["location"])
        return None

    prediction = str(model_result["prediction"])
    confidence = float(model_result.get("confidence", 0))
    prediction_record = {
        "id": str(uuid4()),
        "window_id": f"{device_id}-{uuid4().hex[:12]}",
        "timestamp": utc_now(),
        "device_id": device_id,
        "status": model_result.get("status", "ALERT"),
        "prediction": prediction,
        "displayResult": display_result(prediction),
        "confidence": confidence,
        "probabilities": model_result.get("probabilities", {}),
        "features": model_result.get("features", {}),
        "source_status": window[-1].get("source_status"),
        "test_object": window[-1].get("test_object"),
    }
    predictions[device_id].insert(0, prediction_record)
    del predictions[device_id][100:]

    device.update(
        {
            "status": "ONLINE" if prediction in SAFE_LABELS else "ALERT",
            "lastResult": display_result(prediction),
            "confidence": round(confidence * 100, 2),
            "latestPrediction": prediction_record,
            "lastSync": prediction_record["timestamp"],
        }
    )
    add_log(
        device_id,
        f"{display_result(prediction)} detected - Confidence {confidence * 100:.1f}%",
        "SUCCESS" if prediction in SAFE_LABELS else "ALERT",
        device["location"],
    )

    if prediction not in SAFE_LABELS:
        incident_id = f"{device_id}-{prediction_record['timestamp'].replace(':', '').replace('-', '')[:15]}"
        incidents[incident_id] = {
            "id": incident_id,
            "type": display_result(prediction),
            "modelPrediction": prediction,
            "location": device["location"],
            "platform": device["platform"],
            "confidence": round(confidence * 100, 2),
            "status": "RESPONSE DISPATCHED",
            "team": "SECURITY RESPONSE",
            "time": prediction_record["timestamp"],
            "device": device_id,
            "predictionId": prediction_record["id"],
            "latestReading": device["latestReading"],
        }

    return prediction_record


def process_pending_windows(device_id: str) -> dict[str, Any] | None:
    latest_prediction = None
    queue = pending_windows[device_id]
    while queue:
        prediction = create_prediction(device_id, queue[0])
        if prediction is None:
            break
        queue.popleft()
        latest_prediction = prediction
    return latest_prediction


@app.get("/health")
def health():
    return jsonify({"status": "ok", "ml_server": ML_SERVER_URL})


@app.post("/api/v1/devices/<device_id>/readings")
def ingest_reading(device_id: str):
    if not device_id.strip():
        return error("Device ID is required.")
    reading, validation_error = validate_reading(request.get_json(silent=True))
    if validation_error:
        return error(validation_error)

    with lock:
        device = get_device(device_id)
        device.update(
            {
                "status": "ONLINE",
                "battery": reading.get("battery", device["battery"]),
                "signal": reading.get("signal", device["signal"]),
                "health": 100,
                "lastSync": reading["timestamp"],
                "latestReading": {**reading, "device_id": device_id},
            }
        )
        windows[device_id].append(
            {
                key: reading[key]
                for key in ["timestamp", *READING_FIELDS, *OPTIONAL_METADATA_FIELDS]
                if key in reading
            }
        )
        add_log(device_id, "Sensor reading received", "INFO", device["location"])

        prediction = process_pending_windows(device_id)
        if len(windows[device_id]) == WINDOW_SIZE:
            window = list(windows[device_id])
            windows[device_id].clear()
            pending_windows[device_id].append(window)
            prediction = process_pending_windows(device_id) or prediction

        response = {
            "accepted": True,
            "device_id": device_id,
            "buffer_count": len(windows[device_id]),
            "window_size": WINDOW_SIZE,
            "pending_windows": len(pending_windows[device_id]),
            "prediction": prediction,
            "latestReading": device["latestReading"],
        }
    return jsonify(response), 200


@app.get("/api/v1/devices")
def list_devices():
    with lock:
        return jsonify({"devices": list(devices.values())})


@app.get("/api/v1/devices/<device_id>")
def device_detail(device_id: str):
    with lock:
        if device_id not in devices:
            return error("Device not found.", 404)
        return jsonify(devices[device_id])


@app.get("/api/v1/devices/<device_id>/predictions")
def device_predictions(device_id: str):
    limit = min(max(request.args.get("limit", default=10, type=int), 1), 100)
    with lock:
        if device_id not in devices:
            return error("Device not found.", 404)
        return jsonify({"predictions": predictions[device_id][:limit]})


@app.get("/api/v1/dashboard/summary")
def dashboard_summary():
    with lock:
        values = list(devices.values())
        active = [item for item in incidents.values() if item["status"] not in {"RESOLVED"}]
        return jsonify(
            {
                "connectedDevices": len(values),
                "online": sum(item["status"] != "OFFLINE" for item in values),
                "offline": sum(item["status"] == "OFFLINE" for item in values),
                "activeAlerts": len(active),
                "activeIncidents": active,
            }
        )


@app.get("/api/v1/incidents")
def list_incidents():
    status = request.args.get("status")
    with lock:
        values = list(incidents.values())
        if status == "active":
            values = [item for item in values if item["status"] != "RESOLVED"]
        return jsonify({"incidents": values})


def update_incident(incident_id: str, status: str):
    with lock:
        incident = incidents.get(incident_id)
        if incident is None:
            return error("Incident not found.", 404)
        incident["status"] = status
        operator_id = (request.get_json(silent=True) or {}).get("operator_id", "UNKNOWN")
        incident["operatorId"] = operator_id
        incident["updatedAt"] = utc_now()
        add_log(incident["device"], f"Incident {status.lower()} by {operator_id}", "SUCCESS", incident["location"])
        return jsonify(incident)


@app.post("/api/v1/incidents/<incident_id>/acknowledge")
def acknowledge_incident(incident_id: str):
    return update_incident(incident_id, "ACKNOWLEDGED")


@app.post("/api/v1/incidents/<incident_id>/resolve")
def resolve_incident(incident_id: str):
    return update_incident(incident_id, "RESOLVED")


@app.get("/api/v1/logs")
def list_logs():
    limit = min(max(request.args.get("limit", default=100, type=int), 1), 500)
    with lock:
        return jsonify({"logs": list(logs)[:limit]})


if __name__ == "__main__":
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False)