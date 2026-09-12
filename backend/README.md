# SENTRY backend

The backend accepts one sensor reading at a time, buffers 30 readings per device, and forwards each complete window to the existing ML server. State is intentionally in memory for local development; restarting the process clears device state, predictions, incidents, and logs.

Install dependencies from the repository root:

```powershell
python -m pip install -r backend/requirements.txt
```

Start the ML server first, then start this backend:

```powershell
python ml_test/ml_server.py
python backend/app.py
```

The API listens on `http://127.0.0.1:8000` by default. Configuration is provided through the variables in `.env.example`.