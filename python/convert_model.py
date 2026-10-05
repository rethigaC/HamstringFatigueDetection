from pathlib import Path

MODEL_PATH = Path(
    "models/hamstring_fatigue_tcn_attention_int8.tflite"
)

OUTPUT_PATH = Path(
    "esp32/model_data.cc"
)

# Check model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

# Read model
data = MODEL_PATH.read_bytes()

# Create C++ file
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    f.write("// Auto-generated TFLite model\n")
    f.write("#include <stdint.h>\n\n")

    f.write(
        "alignas(16) const unsigned char "
        "hamstring_fatigue_model[] = {\n"
    )

    for i, byte in enumerate(data):

        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02x}")

        if i != len(data) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n\n")

    f.write(
        "const unsigned int "
        "hamstring_fatigue_model_len = "
        f"{len(data)};\n"
    )

print("=" * 60)
print("TFLITE -> C++ CONVERSION COMPLETED")
print("=" * 60)

print("Input :", MODEL_PATH)
print("Output:", OUTPUT_PATH)
print("Size  :", f"{len(data):,} bytes")