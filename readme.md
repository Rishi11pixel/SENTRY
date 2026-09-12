 # SENTRY

SENTRY is a sensor-monitoring and anomaly-detection project. It combines a React control-center frontend with Python machine-learning workflows for classifying synthetic sensor windows.

## Features

- Live-style control center with dashboard, device inventory, tracking, alerts, threat history, and system logs.
- Dark and light themes with responsive navigation.
- Random Forest and TinyML neural-network classifiers.
- Float32 and int8 TensorFlow Lite model artifacts.
- Evaluation reports and confusion-matrix visuals.

## Repository Layout

```text
data/raw/       Synthetic training and final test datasets
firmware/       Embedded firmware files
Frontend/       React + Vite control-center application
ml/             Dataset, training, conversion, and evaluation scripts
models/         Trained model and preprocessing artifacts
results/        Metrics, metadata, and generated visualizations
```

## Frontend Setup

Requirements: Node.js and pnpm.

```powershell
cd Frontend
pnpm install
pnpm dev
```

Open the local URL printed by Vite. To create a production build:

```powershell
pnpm build
```

The frontend is a prototype control center with backend integration and static data as a fallback when the local backend is unavailable.

## Local End-to-End Run

The integration backend accepts one sensor reading at a time, buffers 30 readings per device, and forwards complete windows to the existing ML server. State is held in memory for local development.

Install backend dependencies and run the services in separate PowerShell terminals:

```powershell
python -m pip install -r backend/requirements.txt
python ml_test/ml_server.py
python backend/app.py
python ml_test/send_readings.py
```

The backend listens on `http://127.0.0.1:8000` and the ML server listens on `http://127.0.0.1:5000`. The frontend reads `VITE_API_URL`, defaulting to the backend URL above. Set it before starting Vite when the backend is hosted elsewhere:

```powershell
$env:VITE_API_URL = "http://127.0.0.1:8000"
cd Frontend
pnpm install
pnpm dev
```

The fake sender uses `SENTRY_BACKEND_URL` when a different ingestion URL is required. The backend settings are listed in `backend/.env.example`.

## Machine-Learning Setup

Requirements: Python 3.10+ and the packages imported by the scripts in `ml/` (including NumPy, pandas, scikit-learn, joblib, matplotlib, and TensorFlow).

Create and activate a virtual environment from the repository root, then install the required packages with your preferred package manager. For example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scikit-learn joblib matplotlib tensorflow
```

Run the workflows from the repository root so generated paths resolve consistently:

```powershell
python ml/generate_dataset.py
python ml/train_random_forest.py
python ml/train_tinyml.py
python ml/quantize_tinyml.py
python ml/convert_tflite.py
python ml/evaluate_final.py
```

The TinyML pipeline uses these seven features:

```text
VMQ2, VMQ3, VMQ135, VSEN0567, dVdt_max, temperature, humidity
```

The canonical fake ESP32 source is `sentry_fake_readings_1800.txt`. The sender
requires that file, parses exactly 1,800 readings, and sends one reading per
request. It does not fall back to `ml_test/sensor_readings.txt` or generate
replacement sensor values. Battery and signal are synthetic device metadata;
they are not ML features.

`VSEN0567` represents the DFRobot SEN0567 NH3-sensitive sensor. The synthetic
generator applies environmental compensation independently to each gas sensor:

```text
V_comp = V_raw - alpha_H * (humidity - 50.0) - alpha_T * (temperature - 25.0)
```

The references are `H_REF = 50.0 %RH` and `T_REF = 25.0 C`. The alpha values
are prototype coefficients for internally consistent synthetic simulation only;
they are not experimentally calibrated constants. Raw sensor behavior includes
event response, environmental distortion, noise, offsets, gain variation, drift,
cross-sensitivity, and ADC quantization before compensation. The resulting data
is synthetic and is not laboratory-calibrated.

The generator uses separate random seeds for the 60,000-sample training set and
the 12,000-sample blind final test set. The final test set is not used for
training, scaler fitting, tuning, quantization calibration, or representative data.

It classifies five labels: `SAFE`, `WEATHER`, `ALCOHOL`, `EXPLOSIVE`, and `NARCOTIC`.

## Final Evaluation

On the untouched 12,000-sample blind test set, the regenerated models measured:

```text
Random Forest:   89.45% accuracy, 89.50% macro F1
TinyML Float32:  87.90% accuracy, 87.87% macro F1
TinyML INT8:     87.70% accuracy, 87.67% macro F1
```

The INT8 model is 3,608 bytes (3.52 KB). Its measured change from TinyML
Float32 was -0.20 percentage points in accuracy and -0.19 percentage points
in macro F1.

The compensation ablation measured Random Forest accuracy of 84.11% without
compensation versus 89.45% with compensation. TinyML Float32 measured 81.67%
without compensation versus 87.90% with compensation.

The weather robustness evaluation covered dry, normal, humid, very humid,
cold, hot, hot+humid, and hot+dry conditions. Random Forest SAFE accuracy
ranged from 37.5% to 96.5% across those synthetic stress conditions, while
WEATHER accuracy ranged from 87.0% to 99.0%. These are synthetic robustness
measurements, not laboratory or field performance claims.

## Generated Artifacts

- `models/random_forest/` contains the Random Forest model and label encoder.
- `models/tinyml/` contains the Keras model, TensorFlow Lite model, label classes, and scaler parameters.
- `results/` contains evaluation JSON, TinyML metadata, and visualizations.

## Notes

- The included datasets are synthetic and should not be treated as evidence of real-world detection performance.
- Model preprocessing parameters must remain paired with the model that produced them.
- Evaluation scripts validate the expected final-test size and class distribution before reporting metrics.
