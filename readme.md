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

The frontend is a prototype control center and currently uses local/static application data rather than a connected backend.

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
VMQ2, VMQ3, VMQ135, VSEN0568, dVdt_max, temperature, humidity
```

It classifies five labels: `AMBIENT_CLEAN`, `WEATHER_DRIFT`, `ALCOHOL_SANITIZER`, `EXPLOSIVE_PROXY`, and `NARCOTIC_PROXY`.

## Generated Artifacts

- `models/random_forest/` contains the Random Forest model and label encoder.
- `models/tinyml/` contains the Keras model, TensorFlow Lite model, label classes, and scaler parameters.
- `results/` contains evaluation JSON, TinyML metadata, and visualizations.

## Notes

- The included datasets are synthetic and should not be treated as evidence of real-world detection performance.
- Model preprocessing parameters must remain paired with the model that produced them.
- Evaluation scripts validate the expected final-test size and class distribution before reporting metrics.
