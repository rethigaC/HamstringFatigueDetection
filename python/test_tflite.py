import numpy as np
import pandas as pd
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/hamstring_fatigue_tcn_attention_int8.tflite"
)

TEST_FILE = "data/processed/test.csv"

TIME_STEPS = 20
FEATURES = 10
TOTAL_FEATURES = 200


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("       INT8 TFLITE MODEL TEST")
print("=" * 60)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

df = pd.read_csv(TEST_FILE)

print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# IDENTIFY LABEL
# ============================================================

label_column = None

for col in df.columns:

    if col.lower() in [
        "fatigue_label",
        "label",
        "fatigue",
        "target"
    ]:
        label_column = col
        break

if label_column is None:

    raise ValueError(
        "Fatigue label column not found."
    )


# ============================================================
# IDENTIFY FEATURES
# ============================================================

excluded = {
    "subject",
    "sequence_id",
    "sequence",
    "id",
    label_column
}

feature_columns = [
    col
    for col in df.columns
    if col not in excluded
]


print("\nFeature columns found:")
print(len(feature_columns))


# ============================================================
# SAFETY CHECK
# ============================================================

if len(feature_columns) != TOTAL_FEATURES:

    print("\nIncorrect feature count.")

    print("\nDetected features:")

    for i, col in enumerate(feature_columns):

        print(
            i,
            ":",
            col
        )

    raise ValueError(
        f"Expected {TOTAL_FEATURES} "
        f"features but found "
        f"{len(feature_columns)}"
    )


# ============================================================
# PREPARE TEST INPUT
# ============================================================

X_test = df[
    feature_columns
].astype(
    np.float32
).values


X_test = X_test.reshape(
    -1,
    TIME_STEPS,
    FEATURES
)


y_test = df[
    label_column
].astype(
    np.int32
).values


print("\nTest input shape:")
print(X_test.shape)

print("\nTest labels:")
print(y_test.shape)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING INT8 TFLITE MODEL")
print("=" * 60)

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()


# ============================================================
# GET INPUT / OUTPUT DETAILS
# ============================================================

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)


input_index = (
    input_details[0]["index"]
)

output_index = (
    output_details[0]["index"]
)


print("\nInput details:")
print(input_details[0])


print("\nOutput details:")
print(output_details[0])


# ============================================================
# QUANTIZATION PARAMETERS
# ============================================================

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)


print("\nInput quantization:")
print(
    "Scale       :",
    input_scale
)

print(
    "Zero point  :",
    input_zero_point
)


print("\nOutput quantization:")
print(
    "Scale       :",
    output_scale
)

print(
    "Zero point  :",
    output_zero_point
)


# ============================================================
# CHECK INT8
# ============================================================

input_dtype = (
    input_details[0]["dtype"]
)

output_dtype = (
    output_details[0]["dtype"]
)


print("\nInput dtype:")
print(input_dtype)

print("\nOutput dtype:")
print(output_dtype)


# ============================================================
# TFLITE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("GENERATING TFLITE PREDICTIONS")
print("=" * 60)


predictions = []


for i in range(
    len(X_test)
):

    sample = X_test[
        i:i + 1
    ]


    # --------------------------------------------------------
    # FLOAT → INT8
    # --------------------------------------------------------

    if input_dtype == np.int8:

        sample_quantized = (
            sample / input_scale
            + input_zero_point
        )

        sample_quantized = np.clip(
            sample_quantized,
            -128,
            127
        ).astype(
            np.int8
        )

    else:

        sample_quantized = sample.astype(
            input_dtype
        )


    # --------------------------------------------------------
    # SET INPUT
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_index,
        sample_quantized
    )


    # --------------------------------------------------------
    # RUN MODEL
    # --------------------------------------------------------

    interpreter.invoke()


    # --------------------------------------------------------
    # GET OUTPUT
    # --------------------------------------------------------

    output = interpreter.get_tensor(
        output_index
    )


    # --------------------------------------------------------
    # DEQUANTIZE
    # --------------------------------------------------------

    if output_dtype == np.int8:

        probability = (
            output.astype(
                np.float32
            ) - output_zero_point
        ) * output_scale

        probability = float(
            probability[0][0]
        )

    else:

        probability = float(
            output[0][0]
        )


    predictions.append(
        probability
    )


# ============================================================
# PREDICTION ARRAY
# ============================================================

probabilities = np.array(
    predictions,
    dtype=np.float32
)


# ============================================================
# CLASSIFICATION
# ============================================================

predicted_labels = (
    probabilities >= 0.5
).astype(
    np.int32
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn = np.sum(
    (y_test == 0) &
    (predicted_labels == 0)
)

fp = np.sum(
    (y_test == 0) &
    (predicted_labels == 1)
)

fn = np.sum(
    (y_test == 1) &
    (predicted_labels == 0)
)

tp = np.sum(
    (y_test == 1) &
    (predicted_labels == 1)
)


# ============================================================
# METRICS
# ============================================================

total = (
    tn +
    fp +
    fn +
    tp
)


accuracy = (
    (tp + tn) /
    total
)


precision = (
    tp /
    (tp + fp)
    if (tp + fp) > 0
    else 0
)


recall = (
    tp /
    (tp + fn)
    if (tp + fn) > 0
    else 0
)


specificity = (
    tn /
    (tn + fp)
    if (tn + fp) > 0
    else 0
)


f1 = (
    2 *
    precision *
    recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("       INT8 TFLITE TEST RESULTS")
print("=" * 60)


print("\nConfusion Matrix:")
print()

print(
    "                 Predicted"
)

print(
    "                 Non-Fatigue   Fatigue"
)

print(
    f"Actual Non-Fatigue "
    f"{tn:10d} "
    f"{fp:11d}"
)

print(
    f"Actual Fatigue     "
    f"{fn:10d} "
    f"{tp:11d}"
)


print("\n" + "=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)

print(
    f"\nAccuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Recall      : {recall:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"F1 Score    : {f1:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame({

    "actual_label":
        y_test,

    "predicted_probability":
        probabilities,

    "predicted_label":
        predicted_labels

})


OUTPUT_FILE = (
    "models/tflite_test_predictions.csv"
)


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nPredictions saved to:")

print(
    OUTPUT_FILE
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("       INT8 TFLITE TEST COMPLETED")
print("=" * 60)