import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    Dropout,
    Add,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Dense
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 20
FEATURES_PER_STEP = 10
TOTAL_FEATURES = SEQUENCE_LENGTH * FEATURES_PER_STEP

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

TRAIN_FILE = "data/processed/train.csv"
VALIDATION_FILE = "data/processed/validation.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "best_hamstring_fatigue_tcn_attention.keras"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "hamstring_fatigue_tcn_attention.keras"
)

HISTORY_FILE = os.path.join(
    MODEL_DIR,
    "attention_training_history.csv"
)

PREDICTIONS_FILE = os.path.join(
    MODEL_DIR,
    "attention_test_predictions.csv"
)

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("       HAMSTRING FATIGUE TCN + SELF-ATTENTION")
print("=" * 60)

print()
print("Input shape:")
print("(samples, 20, 10)")
print()
print("Total features per sequence:", TOTAL_FEATURES)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(file_path):

    print()
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    print()
    print("Reading:", file_path)

    df = pd.read_csv(file_path)

    print()
    print("Rows    :", len(df))
    print("Columns :", len(df.columns))

    # --------------------------------------------------------
    # FIND LABEL COLUMN
    # --------------------------------------------------------

    label_candidates = [
        "fatigue_label",
        "label",
        "target",
        "class"
    ]

    label_column = None

    for col in label_candidates:
        if col in df.columns:
            label_column = col
            break

    if label_column is None:
        raise ValueError(
            "Could not find fatigue label column."
        )

    # --------------------------------------------------------
    # SUBJECT COLUMN
    # --------------------------------------------------------

    subject_column = None

    for col in ["subject", "Subject", "subject_id"]:
        if col in df.columns:
            subject_column = col
            break

    # --------------------------------------------------------
    # SEQUENCE ID
    # --------------------------------------------------------

    sequence_id_column = None

    if "sequence_id" in df.columns:
        sequence_id_column = "sequence_id"

    # --------------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------------

    excluded_columns = {
        label_column
    }

    if subject_column is not None:
        excluded_columns.add(subject_column)

    if sequence_id_column is not None:
        excluded_columns.add(sequence_id_column)

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    print()
    print("CSV columns:", len(df.columns))

    if subject_column:
        print("Subject column :", subject_column)
    else:
        print("Subject column : NOT FOUND")

    if sequence_id_column:
        print("Sequence ID    :", sequence_id_column)

    print("Feature columns:", len(feature_columns))
    print("Label column   :", label_column)

    # --------------------------------------------------------
    # CHECK FEATURE COUNT
    # --------------------------------------------------------

    if len(feature_columns) != TOTAL_FEATURES:

        print()
        print("ERROR: Incorrect feature count.")
        print()
        print("Expected features :", TOTAL_FEATURES)
        print("Found features    :", len(feature_columns))
        print()

        print("Detected feature columns:")

        for i, col in enumerate(feature_columns):
            print(f"{i}: {col}")

        raise ValueError(
            f"Expected {TOTAL_FEATURES} features "
            f"but found {len(feature_columns)}"
        )

    # --------------------------------------------------------
    # CONVERT FEATURES TO NUMERIC
    # --------------------------------------------------------

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    ).values.astype(np.float32)

    y = pd.to_numeric(
        df[label_column],
        errors="coerce"
    ).values.astype(np.int32)

    # --------------------------------------------------------
    # CHECK INVALID VALUES
    # --------------------------------------------------------

    if np.isnan(X).any():

        print()
        print("WARNING: NaN values found in features.")
        print("Replacing NaN values with zero.")

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    if np.isnan(y).any():

        raise ValueError(
            "NaN values found in labels."
        )

    # --------------------------------------------------------
    # RESHAPE
    # --------------------------------------------------------

    X = X.reshape(
        -1,
        SEQUENCE_LENGTH,
        FEATURES_PER_STEP
    )

    print()
    print("Samples :", X.shape[0])
    print("Features:", X.shape[1:])
    print("X shape :", X.shape)
    print("Y shape :", y.shape)

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print()
    print("Label distribution:")

    for label, count in zip(unique, counts):

        if label == 0:
            name = "Non-Fatigue"
        else:
            name = "Fatigue"

        print(
            f"{name} ({label}): {count}"
        )

    return X, y


# ============================================================
# LOAD TRAINING DATA
# ============================================================

X_train, y_train = load_dataset(TRAIN_FILE)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

X_val, y_val = load_dataset(VALIDATION_FILE)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test, y_test = load_dataset(TEST_FILE)


# ============================================================
# VERIFY SHAPES
# ============================================================

print()
print("=" * 60)
print("DATA SHAPES")
print("=" * 60)

print()
print("Training   :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing    :", X_test.shape)


# ============================================================
# TCN RESIDUAL BLOCK
# ============================================================

def tcn_residual_block(
    x,
    filters,
    kernel_size,
    dilation_rate,
    dropout_rate=0.2
):

    # --------------------------------------------------------
    # First causal convolution
    # --------------------------------------------------------

    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate
    )(x)

    conv1 = BatchNormalization()(conv1)

    conv1 = Activation("relu")(conv1)

    conv1 = Dropout(
        dropout_rate
    )(conv1)

    # --------------------------------------------------------
    # Second causal convolution
    # --------------------------------------------------------

    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate
    )(conv1)

    conv2 = BatchNormalization()(conv2)

    conv2 = Activation("relu")(conv2)

    conv2 = Dropout(
        dropout_rate
    )(conv2)

    # --------------------------------------------------------
    # Residual connection
    # --------------------------------------------------------

    if x.shape[-1] != filters:

        residual = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same"
        )(x)

    else:

        residual = x

    output = Add()([
        residual,
        conv2
    ])

    output = Activation("relu")(output)

    return output


# ============================================================
# SELF-ATTENTION BLOCK
# ============================================================

def self_attention_block(
    x,
    num_heads=4,
    key_dim=32,
    dropout_rate=0.1
):

    # --------------------------------------------------------
    # Layer normalization
    # --------------------------------------------------------

    normalized = LayerNormalization()(
        x
    )

    # --------------------------------------------------------
    # Multi-head self-attention
    # --------------------------------------------------------

    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate
    )(
        normalized,
        normalized
    )

    # --------------------------------------------------------
    # Residual connection
    # --------------------------------------------------------

    x = Add()([
        x,
        attention_output
    ])

    # --------------------------------------------------------
    # Feed-forward network
    # --------------------------------------------------------

    normalized_2 = LayerNormalization()(
        x
    )

    ff = Dense(
        128,
        activation="relu"
    )(normalized_2)

    ff = Dropout(
        dropout_rate
    )(ff)

    ff = Dense(
        int(x.shape[-1])
    )(ff)

    # --------------------------------------------------------
    # Second residual connection
    # --------------------------------------------------------

    output = Add()([
        x,
        ff
    ])

    return output


# ============================================================
# BUILD TCN + SELF-ATTENTION MODEL
# ============================================================

def build_model():

    inputs = Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES_PER_STEP
        ),
        name="sEMG_Input"
    )

    # ========================================================
    # TCN BLOCK 1
    # ========================================================

    x = tcn_residual_block(
        inputs,
        filters=32,
        kernel_size=3,
        dilation_rate=1,
        dropout_rate=0.20
    )

    # ========================================================
    # TCN BLOCK 2
    # ========================================================

    x = tcn_residual_block(
        x,
        filters=32,
        kernel_size=3,
        dilation_rate=2,
        dropout_rate=0.20
    )

    # ========================================================
    # TCN BLOCK 3
    # ========================================================

    x = tcn_residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=4,
        dropout_rate=0.20
    )

    # ========================================================
    # TCN BLOCK 4
    # ========================================================

    x = tcn_residual_block(
        x,
        filters=64,
        kernel_size=3,
        dilation_rate=8,
        dropout_rate=0.20
    )

    # ========================================================
    # SELF-ATTENTION
    # ========================================================

    x = self_attention_block(
        x,
        num_heads=4,
        key_dim=16,
        dropout_rate=0.10
    )

    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    x = GlobalAveragePooling1D(
        name="Global_Average_Pooling"
    )(x)

    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = Dense(
        64,
        activation="relu",
        name="Dense_64"
    )(x)

    x = Dropout(
        0.30,
        name="Classification_Dropout"
    )(x)

    outputs = Dense(
        1,
        activation="sigmoid",
        name="Fatigue_Output"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="TCN_SelfAttention_Fatigue_Model"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 60)
print("BUILDING TCN + SELF-ATTENTION MODEL")
print("=" * 60)

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    BEST_MODEL,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger(
    HISTORY_FILE,
    append=False
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("STARTING TCN + SELF-ATTENTION TRAINING")
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

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ],

    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL
)

print()
print("Final model saved to:")
print(FINAL_MODEL)

print()
print("Best model saved to:")
print(BEST_MODEL)


# ============================================================
# TEST MODEL
# ============================================================

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

test_results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    test_results
):

    print(
        f"{name:<15}: {value:.4f}"
    )


# ============================================================
# TEST PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("GENERATING TEST PREDICTIONS")
print("=" * 60)

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print()
print("                 Predicted")
print("                 Non-Fatigue   Fatigue")
print(
    f"Actual Non-Fatigue      {tn:5d}        {fp:5d}"
)
print(
    f"Actual Fatigue          {fn:5d}        {tp:5d}"
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# ============================================================
# PRINT METRICS
# ============================================================

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
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "actual_label": y_test,
    "predicted_label": predictions,
    "fatigue_probability": probabilities
})

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print()
print("Test predictions saved to:")
print(PREDICTIONS_FILE)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("       TCN + SELF-ATTENTION TRAINING COMPLETED")
print("=" * 60)

print()
print("Input shape:")
print("(20 time steps × 10 features)")

print()
print("Training samples :", len(X_train))
print("Validation samples :", len(X_val))
print("Testing samples :", len(X_test))

print()
print("Architecture:")
print("TCN Residual Blocks")
print("        ↓")
print("Multi-Head Self-Attention")
print("        ↓")
print("Global Average Pooling")
print("        ↓")
print("Dense Classification")
print("        ↓")
print("Fatigue / Non-Fatigue")

print()
print("Final model:")
print(FINAL_MODEL)

print()
print("Best model:")
print(BEST_MODEL)

print()
print("Training history:")
print(HISTORY_FILE)

print()
print("Test predictions:")
print(PREDICTIONS_FILE)

print()
print("=" * 60)
print("READY FOR MODEL ANALYSIS")
print("=" * 60)