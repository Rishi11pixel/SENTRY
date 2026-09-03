import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "sentry_synthetic_dataset.csv"
MODEL_DIR = ROOT / "models" / "random_forest"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0568",
    "dVdt_max",
    "temperature",
    "humidity",
]

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y_text = df["label"]

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

print("Classes:")
for number, name in enumerate(encoder.classes_):
    print(f"  {number}: {name}")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print()
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

print(f"\nValidation accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        target_names=encoder.classes_,
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_val,
        predictions,
    )
)

print("\nFeature Importance:")

importance = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)

for feature, value in importance:
    print(f"{feature:15s}: {value:.4f}")

joblib.dump(
    model,
    MODEL_DIR / "random_forest.joblib",
)

joblib.dump(
    encoder,
    MODEL_DIR / "label_encoder.joblib",
)

print("\nSaved:")
print(MODEL_DIR / "random_forest.joblib")
print(MODEL_DIR / "label_encoder.joblib")