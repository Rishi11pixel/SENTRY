# ml/train_tinyml.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.csv"

MODEL_DIR = BASE_DIR / "models" / "tinyml"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

KERAS_MODEL_PATH = MODEL_DIR / "sentry_tinyml.keras"
TFLITE_MODEL_PATH = MODEL_DIR / "sentry_tinyml_int8.tflite"

SCALER_MEAN_PATH = MODEL_DIR / "scaler_mean.npy"
SCALER_SCALE_PATH = MODEL_DIR / "scaler_scale.npy"

LABELS_PATH = MODEL_DIR / "labels.json"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "dVdt_max",
    "temperature",
    "humidity",
]

LABELS = [
    "SAFE",
    "WEATHER",
    "ALCOHOL",
    "EXPLOSIVE",
    "NARCOTIC",
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    i: label
    for i, label in enumerate(LABELS)
}


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

EPOCHS = 60
BATCH_SIZE = 128

LEARNING_RATE = 0.001


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 60)
print("             SENTRY TinyML TRAINING")
print("=" * 60)

print()
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")

# ------------------------------------------------------------
# Validate dataset
# ------------------------------------------------------------

required_columns = FEATURES + ["label"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Make sure SEN0567 is completely gone
for column in df.columns:
    if "SEN0567" in column.upper():
        raise ValueError(
            f"SEN0567 still exists in dataset: {column}"
        )

# Make sure labels are valid
unknown_labels = set(df["label"].unique()) - set(LABELS)

if unknown_labels:
    raise ValueError(
        f"Unknown labels found: {unknown_labels}"
    )


# ============================================================
# PREPARE X / Y
# ============================================================

X = df[FEATURES].astype(np.float32).values

y = np.array([
    LABEL_TO_ID[label]
    for label in df["label"]
], dtype=np.int32)


print()
print("Features:")
for i, feature in enumerate(FEATURES):
    print(f"  {i} -> {feature}")

print()
print("Classes:")
for i, label in enumerate(LABELS):
    print(f"  {i} -> {label}")

print()
print(f"Input feature count: {X.shape[1]}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    stratify=y_train_full,
)

print()
print("Dataset split:")
print(f"  Training   : {X_train.shape}")
print(f"  Validation : {X_val.shape}")
print(f"  Test       : {X_test.shape}")


# ============================================================
# STANDARD SCALER
# ============================================================

print()
print("Fitting StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Save scaler parameters
np.save(
    SCALER_MEAN_PATH,
    scaler.mean_.astype(np.float32),
)

np.save(
    SCALER_SCALE_PATH,
    scaler.scale_.astype(np.float32),
)

print("Scaler saved.")


# ============================================================
# SAVE LABELS
# ============================================================

with open(LABELS_PATH, "w") as f:
    json.dump(LABELS, f, indent=2)

print("Labels saved.")


# ============================================================
# BUILD TINYML MODEL
# ============================================================

print()
print("Building TinyML neural network...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(len(FEATURES),),
        name="input",
    ),

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
        len(LABELS),
        activation="softmax",
        name="classification",
    ),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# TRAINING
# ============================================================

print()
print("Training...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(
        X_val_scaled,
        y_val,
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1,
)


# ============================================================
# SAVE KERAS MODEL
# ============================================================

model.save(KERAS_MODEL_PATH)

print()
print(f"Keras model saved:")
print(KERAS_MODEL_PATH)


# ============================================================
# FLOAT MODEL EVALUATION
# ============================================================

print()
print("=" * 60)
print("FLOAT MODEL EVALUATION")
print("=" * 60)

test_probabilities = model.predict(
    X_test_scaled,
    verbose=0,
)

y_pred = np.argmax(
    test_probabilities,
    axis=1,
)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
)

print()
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(LABELS))),
)

print()

header = "             " + " ".join(
    f"{label:>10}"
    for label in LABELS
)

print(header)

for i, label in enumerate(LABELS):
    row = " ".join(
        f"{value:10d}"
        for value in cm[i]
    )

    print(
        f"{label:>10} {row}"
    )


# ============================================================
# INT8 QUANTIZATION
# ============================================================

print()
print("=" * 60)
print("INT8 QUANTIZATION")
print("=" * 60)

print()
print("Converting to INT8 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]


# Representative dataset
def representative_dataset():
    # Use a subset to keep conversion fast
    count = min(1000, len(X_train_scaled))

    for i in range(count):
        sample = X_train_scaled[i:i + 1].astype(
            np.float32
        )

        yield [sample]


converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print()
print("INT8 model saved:")
print(TFLITE_MODEL_PATH)

print()
print(
    f"INT8 model size: "
    f"{len(tflite_model)} bytes"
)


# ============================================================
# INT8 MODEL VERIFICATION
# ============================================================

print()
print("=" * 60)
print("INT8 MODEL VERIFICATION")
print("=" * 60)

interpreter = tf.lite.Interpreter(
    model_path=str(TFLITE_MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)

print()
print("Input:")
print(
    f"  shape      : "
    f"{input_details[0]['shape']}"
)

print(
    f"  dtype      : "
    f"{input_details[0]['dtype']}"
)

print(
    f"  scale      : "
    f"{input_scale}"
)

print(
    f"  zero point : "
    f"{input_zero_point}"
)

print()
print("Output:")
print(
    f"  shape      : "
    f"{output_details[0]['shape']}"
)

print(
    f"  dtype      : "
    f"{output_details[0]['dtype']}"
)

print(
    f"  scale      : "
    f"{output_scale}"
)

print(
    f"  zero point : "
    f"{output_zero_point}"
)


# ============================================================
# INT8 TEST SET EVALUATION
# ============================================================

print()
print("Evaluating INT8 model...")

int8_predictions = []

for sample in X_test_scaled:

    sample = sample.astype(np.float32)

    # Quantize input
    if input_scale == 0:
        raise ValueError(
            "Invalid INT8 input scale."
        )

    quantized = np.round(
        sample / input_scale
        + input_zero_point
    ).astype(np.int8)

    quantized = quantized.reshape(
        1,
        len(FEATURES),
    )

    interpreter.set_tensor(
        input_index,
        quantized,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )[0]

    # Dequantize output
    if output_scale != 0:
        output = (
            output.astype(np.float32)
            - output_zero_point
        ) * output_scale

    prediction = int(
        np.argmax(output)
    )

    int8_predictions.append(
        prediction
    )


int8_predictions = np.array(
    int8_predictions
)


# ============================================================
# INT8 METRICS
# ============================================================

int8_accuracy = accuracy_score(
    y_test,
    int8_predictions,
)

int8_macro_f1 = f1_score(
    y_test,
    int8_predictions,
    average="macro",
)

print()
print(
    f"INT8 Accuracy : "
    f"{int8_accuracy:.4f}"
)

print(
    f"INT8 Macro F1 : "
    f"{int8_macro_f1:.4f}"
)

print()
print("INT8 classification report:")

print(
    classification_report(
        y_test,
        int8_predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
)

print("INT8 confusion matrix:")

int8_cm = confusion_matrix(
    y_test,
    int8_predictions,
    labels=list(range(len(LABELS))),
)

print()

print(header)

for i, label in enumerate(LABELS):

    row = " ".join(
        f"{value:10d}"
        for value in int8_cm[i]
    )

    print(
        f"{label:>10} {row}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()
print(f"Features : {len(FEATURES)}")
print(f"Classes  : {len(LABELS)}")

print()
print(f"Float accuracy : {accuracy:.4f}")
print(f"Float Macro F1 : {macro_f1:.4f}")

print()
print(f"INT8 accuracy  : {int8_accuracy:.4f}")
print(f"INT8 Macro F1  : {int8_macro_f1:.4f}")

print()
print("Artifacts:")

print(f"  Keras : {KERAS_MODEL_PATH}")
print(f"  TFLite: {TFLITE_MODEL_PATH}")
print(f"  Mean  : {SCALER_MEAN_PATH}")
print(f"  Scale : {SCALER_SCALE_PATH}")
print(f"  Labels: {LABELS_PATH}")

print()
print("SEN0567: REMOVED")
print()