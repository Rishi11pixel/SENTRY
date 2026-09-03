import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# Configuration
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "sentry_synthetic_dataset.csv"
)

MODEL_DIR = ROOT / "models" / "tinyml"

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

# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].values.astype(np.float32)
y_text = df["label"].values

# ============================================================
# Encode labels
# ============================================================

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

print("Classes:")

for i, name in enumerate(encoder.classes_):
    print(f"  {i}: {name}")

# ============================================================
# Train/validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

# ============================================================
# Normalize features using only training data
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)

# ============================================================
# Build tiny neural network
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(7,)),

    tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_16",
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu",
        name="dense_8",
    ),

    tf.keras.layers.Dense(
        5,
        activation="softmax",
        name="output",
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nModel:")
model.summary()

# ============================================================
# Train
# ============================================================

print("\nTraining TinyML model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=1,
)

# ============================================================
# Evaluate
# ============================================================

validation_loss, validation_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0,
)

probabilities = model.predict(
    X_val,
    verbose=0,
)

predictions = np.argmax(probabilities, axis=1)

print("\n" + "=" * 70)
print("TINYML FLOAT32 RESULTS")
print("=" * 70)

print(f"\nValidation accuracy: {validation_accuracy:.4f}")

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

# ============================================================
# Save model
# ============================================================

keras_path = MODEL_DIR / "sentry_tinyml.keras"

model.save(keras_path)

print("\nSaved:")
print(keras_path)

# ============================================================
# Save preprocessing parameters
# ============================================================

np.save(
    MODEL_DIR / "scaler_mean.npy",
    scaler.mean_,
)

np.save(
    MODEL_DIR / "scaler_scale.npy",
    scaler.scale_,
)

np.save(
    MODEL_DIR / "label_classes.npy",
    encoder.classes_,
)

print("Saved preprocessing parameters.")