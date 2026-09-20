import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# CONFIG
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURES = [
    "VMQ2",
    "VMQ3",
    "VMQ135",
    "VSEN0567",
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

TRAIN_PATH = "data/raw/sentry_synthetic_dataset.csv"
TEST_PATH = "data/raw/sentry_synthetic_test.csv"

MODEL_DIR = "models/tinyml"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("             SENTRY TinyML TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(TRAIN_PATH)
blind_test = pd.read_csv(TEST_PATH)

print(f"Training dataset: {df.shape}")
print(f"Blind test set  : {blind_test.shape}")


# ============================================================
# FEATURES / LABELS
# ============================================================

X = df[FEATURES].astype(np.float32)

y_text = df["label"].astype(str)

X_test = blind_test[FEATURES].astype(np.float32)
y_test_text = blind_test["label"].astype(str)


# ============================================================
# LABEL ENCODING
# ============================================================

label_to_index = {
    label: i
    for i, label in enumerate(LABELS)
}

index_to_label = {
    i: label
    for label, i in label_to_index.items()
}

y = np.array(
    [
        label_to_index[label]
        for label in y_text
    ],
    dtype=np.int32
)

y_test = np.array(
    [
        label_to_index[label]
        for label in y_test_text
    ],
    dtype=np.int32
)


print("\nClasses:")

for i, label in enumerate(LABELS):
    print(f"  {i} -> {label}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)


# ============================================================
# STANDARD SCALER
# ============================================================
#
# IMPORTANT:
# Fit ONLY on training data.
# ============================================================

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


# ============================================================
# SAVE SCALER
# ============================================================

np.save(
    os.path.join(
        MODEL_DIR,
        "scaler_mean.npy"
    ),
    scaler.mean_.astype(np.float32)
)

np.save(
    os.path.join(
        MODEL_DIR,
        "scaler_scale.npy"
    ),
    scaler.scale_.astype(np.float32)
)


# ============================================================
# SAVE LABELS
# ============================================================

np.save(
    os.path.join(
        MODEL_DIR,
        "label_classes.npy"
    ),
    np.array(
        LABELS,
        dtype=object
    )
)


# ============================================================
# BUILD TINYML MODEL
# ============================================================

print("\nBuilding TinyML neural network...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(7,),
        name="sensor_features"
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_16"
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu",
        name="dense_8"
    ),

    tf.keras.layers.Dense(
        5,
        activation="softmax",
        name="classification"
    )
])


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nTraining...")

history = model.fit(

    X_train_scaled,
    y_train,

    validation_data=(
        X_val_scaled,
        y_val
    ),

    epochs=60,

    batch_size=128,

    verbose=1,

    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]
)


# ============================================================
# FLOAT32 TEST
# ============================================================

print()
print("=" * 60)
print("             FLOAT32 TEST")
print("=" * 60)

test_probabilities = model.predict(
    X_test_scaled,
    verbose=0
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1 : {macro_f1:.4f}"
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=LABELS,
        digits=4
    )
)


# ============================================================
# SAVE FLOAT32 MODEL
# ============================================================

keras_path = os.path.join(
    MODEL_DIR,
    "sentry_tinyml.keras"
)

model.save(
    keras_path
)

print(
    f"\nSaved Float32 model:"
    f"\n  {keras_path}"
)


# ============================================================
# INT8 QUANTIZATION
# ============================================================

print()
print("=" * 60)
print("             INT8 QUANTIZATION")
print("=" * 60)


# Representative dataset
#
# IMPORTANT:
# Use training data only.
# ============================================================

representative_data = X_train_scaled


def representative_dataset():

    # Limit representative samples to keep
    # conversion reasonably fast.

    count = min(
        500,
        len(representative_data)
    )

    for i in range(count):

        sample = representative_data[i]

        sample = sample.reshape(
            1,
            7
        ).astype(np.float32)

        yield [sample]


converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()


# ============================================================
# SAVE INT8 MODEL
# ============================================================

tflite_path = os.path.join(
    MODEL_DIR,
    "sentry_tinyml_int8.tflite"
)

with open(
    tflite_path,
    "wb"
) as f:

    f.write(
        tflite_model
    )


print(
    f"\nSaved INT8 model:"
    f"\n  {tflite_path}"
)

print(
    f"\nINT8 model size:"
    f" {len(tflite_model)} bytes"
)


# ============================================================
# VERIFY TFLITE MODEL
# ============================================================

print()
print("=" * 60)
print("             INT8 MODEL CHECK")
print("=" * 60)

interpreter = tf.lite.Interpreter(
    model_path=tflite_path
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print(
    "  shape :",
    input_details[0]["shape"]
)
print(
    "  dtype :",
    input_details[0]["dtype"]
)
print(
    "  scale :",
    input_details[0]["quantization"][0]
)
print(
    "  zero  :",
    input_details[0]["quantization"][1]
)

print("\nOutput:")
print(
    "  shape :",
    output_details[0]["shape"]
)
print(
    "  dtype :",
    output_details[0]["dtype"]
)
print(
    "  scale :",
    output_details[0]["quantization"][0]
)
print(
    "  zero  :",
    output_details[0]["quantization"][1]
)


# ============================================================
# INT8 BLIND TEST
# ============================================================

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)


correct = 0

int8_predictions = []


print("\nRunning INT8 blind test...")


for i in range(
    len(X_test_scaled)
):

    sample = X_test_scaled[i]

    # Float32 -> INT8
    quantized_input = np.round(
        sample / input_scale
        + input_zero_point
    )

    quantized_input = np.clip(
        quantized_input,
        -128,
        127
    ).astype(np.int8)

    quantized_input = (
        quantized_input.reshape(
            1,
            7
        )
    )

    interpreter.set_tensor(
        input_details[0]["index"],
        quantized_input
    )

    interpreter.invoke()

    quantized_output = (
        interpreter.get_tensor(
            output_details[0]["index"]
        )
    )

    output_float = (
        quantized_output.astype(
            np.float32
        )
        - output_zero_point
    ) * output_scale

    prediction = int(
        np.argmax(
            output_float[0]
        )
    )

    int8_predictions.append(
        prediction
    )

    if prediction == y_test[i]:
        correct += 1


int8_accuracy = (
    correct /
    len(y_test)
)


print(
    f"\nINT8 Accuracy : "
    f"{int8_accuracy * 100:.2f}%"
)


print()
print("=" * 60)
print("             TRAINING COMPLETE")
print("=" * 60)

print("\nGenerated files:")

print(
    f"  {keras_path}"
)

print(
    f"  {tflite_path}"
)

print(
    f"  {MODEL_DIR}/scaler_mean.npy"
)

print(
    f"  {MODEL_DIR}/scaler_scale.npy"
)

print(
    f"  {MODEL_DIR}/label_classes.npy"
)

print()
print("Next step:")
print("Test the INT8 model before ESP32 deployment.")
print("=" * 60)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    int8_predictions
)

print()
print("=" * 60)
print("             INT8 CONFUSION MATRIX")
print("=" * 60)

print(
    "\nRows = TRUE CLASS"
    "\nColumns = PREDICTED CLASS\n"
)

print(
    "             "
    + " ".join(
        f"{label:>10}"
        for label in LABELS
    )
)

for i, label in enumerate(LABELS):

    print(
        f"{label:>10} "
        + " ".join(
            f"{cm[i, j]:10d}"
            for j in range(len(LABELS))
        )
    )


# ============================================================
# SAFE MISCLASSIFICATION BREAKDOWN
# ============================================================

safe_total = np.sum(
    y_test == label_to_index["SAFE"]
)

print()
print("=" * 60)
print("          SAFE MISCLASSIFICATION")
print("=" * 60)

for j, label in enumerate(LABELS):

    count = cm[
        label_to_index["SAFE"],
        j
    ]

    percentage = (
        count / safe_total * 100
    )

    print(
        f"SAFE -> {label:10s}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )