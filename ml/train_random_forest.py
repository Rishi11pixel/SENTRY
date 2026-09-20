import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0567",
    "dVdt_max",
    "temperature",
    "humidity",
]

LABEL_COLUMN = "label"

TRAIN_PATH = "data/raw/sentry_synthetic_dataset.csv"
TEST_PATH = "data/raw/sentry_synthetic_test.csv"

MODEL_DIR = "models/random_forest"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(TRAIN_PATH)
blind_test = pd.read_csv(TEST_PATH)

X = df[FEATURES].astype(np.float32)
y = df[LABEL_COLUMN].astype(str)

X_test = blind_test[FEATURES].astype(np.float32)
y_test = blind_test[LABEL_COLUMN].astype(str)


# ---------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------
# SCALER
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
).astype(np.float32)

X_val_scaled = scaler.transform(
    X_val
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train_scaled,
    y_train,
)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

val_pred = model.predict(X_val_scaled)

print("\n===== VALIDATION =====")
print(
    "Accuracy:",
    accuracy_score(y_val, val_pred)
)
print(
    "Macro F1:",
    f1_score(
        y_val,
        val_pred,
        average="macro"
    )
)

print(
    classification_report(
        y_val,
        val_pred,
        digits=4
    )
)


# ---------------------------------------------------------
# BLIND FINAL TEST
# ---------------------------------------------------------

test_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    test_pred
)

macro_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)

print("\n===== BLIND TEST =====")
print(
    "Accuracy:",
    accuracy
)
print(
    "Macro F1:",
    macro_f1
)

print(
    classification_report(
        y_test,
        test_pred,
        digits=4
    )
)


# ---------------------------------------------------------
# THREAT METRICS
# ---------------------------------------------------------

THREAT = {
    "EXPLOSIVE",
    "NARCOTIC",
}

actual_threat = np.array([
    label in THREAT
    for label in y_test
])

predicted_threat = np.array([
    label in THREAT
    for label in test_pred
])

TP = int(
    np.sum(
        actual_threat &
        predicted_threat
    )
)

TN = int(
    np.sum(
        (~actual_threat) &
        (~predicted_threat)
    )
)

FP = int(
    np.sum(
        (~actual_threat) &
        predicted_threat
    )
)

FN = int(
    np.sum(
        actual_threat &
        (~predicted_threat)
    )
)

threat_recall = (
    TP / (TP + FN)
    if TP + FN > 0
    else 0.0
)

threat_fnr = (
    FN / (TP + FN)
    if TP + FN > 0
    else 0.0
)

print("\n===== THREAT METRICS =====")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print("Threat Recall:", threat_recall)
print("Threat FNR:", threat_fnr)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

print("\n===== CONFUSION MATRIX =====")

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=labels,
)

print("Labels:", labels)
print(cm)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "random_forest.joblib"
    ),
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.joblib"
    ),
)

joblib.dump(
    labels,
    os.path.join(
        MODEL_DIR,
        "label_encoder.joblib"
    ),
)

# Also save scaler parameters in the format
# used by the existing TinyML/inference pipeline.

np.save(
    "models/tinyml/scaler_mean.npy",
    scaler.mean_.astype(np.float32),
)

np.save(
    "models/tinyml/scaler_scale.npy",
    scaler.scale_.astype(np.float32),
)

print("\nSaved:")
print(
    "models/random_forest/random_forest.joblib"
)
print(
    "models/random_forest/scaler.joblib"
)
print(
    "models/random_forest/label_encoder.joblib"
)
print(
    "models/tinyml/scaler_mean.npy"
)
print(
    "models/tinyml/scaler_scale.npy"
)