import tensorflow as tf
import numpy as np
import pandas as pd

MODEL_PATH = "models/best_hamstring_fatigue_tcn_attention.keras"
OUTPUT_PATH = "models/hamstring_fatigue_tcn_attention_int8.tflite"

TRAIN_FILE = "data/processed/train.csv"

TIME_STEPS = 20
FEATURES = 10


print("=" * 60)
print("TFLITE INT8 CONVERSION")
print("=" * 60)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("\nLoading best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

print("\nInput shape:")
print(model.input_shape)

# ------------------------------------------------------------
# REPRESENTATIVE DATA
# ------------------------------------------------------------

print("\nLoading representative dataset...")

df = pd.read_csv(TRAIN_FILE)

# Remove non-feature columns
exclude = [
    "subject",
    "sequence_id",
    "fatigue_label"
]

feature_columns = [
    c for c in df.columns
    if c not in exclude
]

# Keep exactly 200 features
feature_columns = feature_columns[:200]

X = df[feature_columns].astype(
    np.float32
).values

X = X.reshape(
    -1,
    TIME_STEPS,
    FEATURES
)

print("Representative data shape:")
print(X.shape)

# ------------------------------------------------------------
# REPRESENTATIVE DATASET
# ------------------------------------------------------------

def representative_dataset():

    limit = min(
        200,
        len(X)
    )

    for i in range(limit):

        sample = X[i:i+1]

        yield [
            sample.astype(
                np.float32
            )
        ]

# ------------------------------------------------------------
# CONVERTER
# ------------------------------------------------------------

print("\nCreating TFLite converter...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

# Enable optimizations
converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# Representative dataset
converter.representative_dataset = (
    representative_dataset
)

# Force INT8 operations
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# INT8 input/output
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# ------------------------------------------------------------
# CONVERT
# ------------------------------------------------------------

print("\nConverting model to INT8...")

tflite_model = converter.convert()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

with open(
    OUTPUT_PATH,
    "wb"
) as f:

    f.write(
        tflite_model
    )

print("\nConversion successful!")

print("\nSaved model:")
print(OUTPUT_PATH)

print("\nModel size:")
print(
    round(
        len(tflite_model) / 1024,
        2
    ),
    "KB"
)

print("\n" + "=" * 60)
print("INT8 TFLITE CONVERSION COMPLETED")
print("=" * 60)