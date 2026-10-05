import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# CONFIGURATION
# ============================================================

TIME_STEPS = 20
FEATURES = 10
TOTAL_FEATURES = 200

BATCH_SIZE = 32
EPOCHS = 150

LEARNING_RATE = 0.0003

TRAIN_FILE = "data/processed/train.csv"
VAL_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL = (
    "models/best_hamstring_fatigue_tcn_attention_tuned.keras"
)

FINAL_MODEL = (
    "models/hamstring_fatigue_tcn_attention_tuned.keras"
)

HISTORY_FILE = (
    "models/tuned_attention_training_history.csv"
)

PREDICTION_FILE = (
    "models/tuned_attention_test_predictions.csv"
)

# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("   TCN + SELF-ATTENTION FINE-TUNED TRAINING")
print("=" * 60)

print()
print("Input shape:")
print("(samples, 20, 10)")

print()
print("Total features per sequence:", TOTAL_FEATURES)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(file_path):

    print()
    print("=" * 60)
    print("LOADING:", file_path)
    print("=" * 60)

    df = pd.read_csv(file_path)

    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    # --------------------------------------------------------
    # SUBJECT
    # --------------------------------------------------------

    subject_column = None

    for col in df.columns:
        if col.lower() == "subject":
            subject_column = col
            break

    if subject_column is None:
        raise ValueError("Subject column not found.")

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

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
        raise ValueError("Fatigue label column not found.")

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    excluded = {
        subject_column,
        label_column
    }

    for col in df.columns:

        if col.lower() in [
            "sequence_id",
            "sequence",
            "id"
        ]:
            excluded.add(col)

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded
    ]

    print()
    print("Subject column :", subject_column)
    print("Label column   :", label_column)
    print("Feature count  :", len(feature_columns))

    if len(feature_columns) != TOTAL_FEATURES:

        print()
        print("Detected feature columns:")

        for i, col in enumerate(feature_columns):
            print(i, ":", col)

        raise ValueError(
            f"Expected {TOTAL_FEATURES} features "
            f"but found {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    X = df[
        feature_columns
    ].astype(
        np.float32
    ).values

    X = X.reshape(
        -1,
        TIME_STEPS,
        FEATURES
    )

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    y = df[
        label_column
    ].astype(
        np.int32
    ).values

    print()
    print("X shape:", X.shape)
    print("Y shape:", y.shape)

    print()
    print("Class distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        if label == 0:
            name = "Non-Fatigue"
        else:
            name = "Fatigue"

        print(
            f"{name} ({label}): {count}"
        )

    return X, y


# ============================================================
# LOAD DATA
# ============================================================

X_train, y_train = load_dataset(
    TRAIN_FILE
)

X_val, y_val = load_dataset(
    VAL_FILE
)

X_test, y_test = load_dataset(
    TEST_FILE
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

# IMPORTANT:
# Calculated ONLY from training data.

class_counts = np.bincount(
    y_train
)

total_train = len(y_train)

class_weight = {}

for cls in range(
    len(class_counts)
):

    class_weight[cls] = (
        total_train /
        (
            len(class_counts) *
            class_counts[cls]
        )
    )

print()
print("=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

for cls, weight in class_weight.items():

    print(
        f"Class {cls}: {weight:.4f}"
    )


# ============================================================
# TCN BLOCK
# ============================================================

def tcn_block(
    x,
    filters,
    dilation,
    dropout_rate
):

    shortcut = x

    # --------------------------------------------------------
    # Conv 1
    # --------------------------------------------------------

    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="same",
        dilation_rate=dilation,
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    x = layers.SpatialDropout1D(
        dropout_rate
    )(x)

    # --------------------------------------------------------
    # Conv 2
    # --------------------------------------------------------

    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="same",
        dilation_rate=dilation,
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    # --------------------------------------------------------
    # Residual projection
    # --------------------------------------------------------

    if shortcut.shape[-1] != filters:

        shortcut = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(
            shortcut
        )

    # --------------------------------------------------------
    # Residual
    # --------------------------------------------------------

    x = layers.Add()([
        x,
        shortcut
    ])

    x = layers.Activation(
        "relu"
    )(x)

    return x


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    inputs = layers.Input(
        shape=(
            TIME_STEPS,
            FEATURES
        ),
        name="sEMG_Input"
    )

    # ========================================================
    # INITIAL PROJECTION
    # ========================================================

    x = layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="Initial_Conv"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    # ========================================================
    # TCN
    # ========================================================

    x = tcn_block(
        x,
        filters=32,
        dilation=1,
        dropout_rate=0.10
    )

    x = tcn_block(
        x,
        filters=32,
        dilation=2,
        dropout_rate=0.10
    )

    x = tcn_block(
        x,
        filters=64,
        dilation=4,
        dropout_rate=0.15
    )

    x = tcn_block(
        x,
        filters=64,
        dilation=8,
        dropout_rate=0.15
    )

    # ========================================================
    # SELF ATTENTION
    # ========================================================

    attention = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=32,
        dropout=0.05,
        name="Self_Attention"
    )(
        query=x,
        key=x,
        value=x
    )

    # ========================================================
    # ATTENTION RESIDUAL
    # ========================================================

    x = layers.Add(
        name="Attention_Residual"
    )([
        x,
        attention
    ])

    x = layers.LayerNormalization(
        epsilon=1e-6,
        name="Attention_LayerNorm"
    )(x)

    # ========================================================
    # FEED FORWARD
    # ========================================================

    ff = layers.Dense(
        128,
        activation="relu"
    )(x)

    ff = layers.Dropout(
        0.15
    )(ff)

    ff = layers.Dense(
        64
    )(ff)

    # ========================================================
    # FF RESIDUAL
    # ========================================================

    x = layers.Add(
        name="FeedForward_Residual"
    )([
        x,
        ff
    ])

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    # ========================================================
    # GLOBAL POOLING
    # ========================================================

    x = layers.GlobalAveragePooling1D(
        name="Global_Average_Pooling"
    )(x)

    # ========================================================
    # CLASSIFIER
    # ========================================================

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.20
    )(x)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.10
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="Fatigue_Output"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="TCN_SelfAttention_Tuned"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 60)
print("BUILDING FINE-TUNED MODEL")
print("=" * 60)

model = build_model()

model.summary()


# ============================================================
# COMPILE
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    clipnorm=1.0
)

model.compile(

    optimizer=optimizer,

    loss="binary_crossentropy",

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    ModelCheckpoint(
        BEST_MODEL,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-7,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=18,
        restore_best_weights=True,
        verbose=1
    )
]


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING FINE-TUNED TRAINING")
print("=" * 60)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weight,

    callbacks=callbacks,

    shuffle=True,

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL
)

print()
print("Final model saved:")
print(FINAL_MODEL)

print()
print("Best model saved:")
print(BEST_MODEL)


# ============================================================
# SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    HISTORY_FILE,
    index=False
)

print()
print("History saved:")
print(HISTORY_FILE)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

best_model = tf.keras.models.load_model(
    BEST_MODEL
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

results = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

for name, value in zip(
    best_model.metrics_names,
    results
):

    print(
        f"{name:<15}: {value:.4f}"
    )


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

probabilities = best_model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()

predictions = (
    probabilities >= 0.5
).astype(
    np.int32
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn = np.sum(
    (y_test == 0) &
    (predictions == 0)
)

fp = np.sum(
    (y_test == 0) &
    (predictions == 1)
)

fn = np.sum(
    (y_test == 1) &
    (predictions == 0)
)

tp = np.sum(
    (y_test == 1) &
    (predictions == 1)
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()
print("                 Predicted")
print("                 Non-Fatigue   Fatigue")

print(
    f"Actual Non-Fatigue "
    f"{tn:10d} {fp:11d}"
)

print(
    f"Actual Fatigue     "
    f"{fn:10d} {tp:11d}"
)


# ============================================================
# METRICS
# ============================================================

total = (
    tn + fp + fn + tp
)

accuracy = (
    (tp + tn) / total
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


print()
print("=" * 60)
print("CLASSIFICATION METRICS")
print("=" * 60)

print()
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "actual_label": y_test,

    "predicted_probability": probabilities,

    "predicted_label": predictions

})

prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)

print()
print("Predictions saved:")
print(PREDICTION_FILE)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("   FINE-TUNED TCN + SELF-ATTENTION COMPLETED")
print("=" * 60)

print()
print("Train samples      :", len(X_train))
print("Validation samples :", len(X_val))
print("Test samples       :", len(X_test))

print()
print("Best model:")
print(BEST_MODEL)

print()
print("============================================================")
print("COMPARE WITH YOUR TCN-ONLY BASELINE: 98.34%")
print("============================================================")