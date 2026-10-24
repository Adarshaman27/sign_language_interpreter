"""
Rebuild the exact training architecture, load the trained .h5 weights,
and save a single clean .keras file that tensorflowjs_converter can
turn into model.json + weights for the browser.

Usage:
    python convert_model.py
Reads:  isl_model.h5, labels.json
Writes: isl_model_clean.keras
"""

import tensorflow as tf
import json

# -----------------------------
# Model configuration
# -----------------------------
SEQUENCE_LEN = 20
FEATURE_DIM = 126

# Read labels
with open("labels.json", "r") as f:
    labels = json.load(f)

NUM_CLASSES = len(labels)

print("Number of classes:", NUM_CLASSES)

# -----------------------------
# Recreate EXACT architecture
# from train.py
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(SEQUENCE_LEN, FEATURE_DIM)
    ),

    tf.keras.layers.Conv1D(
        64,
        3,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.Conv1D(
        64,
        3,
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# -----------------------------
# Build model
# -----------------------------
model.build((None, SEQUENCE_LEN, FEATURE_DIM))

print("\nModel architecture:")
model.summary()

# -----------------------------
# Load existing trained weights
# -----------------------------
print("\nLoading trained weights...")

model.load_weights("isl_model.h5")

print("Trained weights loaded successfully!")

# -----------------------------
# Save as a clean Keras model
# -----------------------------
output_path = "isl_model_clean.keras"

model.save(output_path)

print("\nComplete model saved:")
print(output_path)

# -----------------------------
# Save labels again
# -----------------------------
with open("labels.json", "w") as f:
    json.dump(labels, f, indent=2)

print("Labels saved.")
print("\nNext: tensorflowjs_converter --input_format keras isl_model_clean.keras ../frontend/model")
