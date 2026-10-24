"""
Train an ISL sign classifier from isl_dataset.json (produced by
data-collector/index.html) into a model recognizer/app.js can load.

IMPORTANT — keep these in sync with recognizer/app.js:
  SEQUENCE_LEN below must equal SEQUENCE_LENGTH in app.js (currently 20).
  If you ever change one, change the other and retrain.

Usage:
    python train.py
Reads:  isl_dataset.json   (place it in this training/ folder)
Writes: isl_model.h5, labels.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SEQUENCE_LEN = 20   # must match SEQUENCE_LENGTH in recognizer/app.js
FEATURE_DIM = 126   # 2 hands x 21 landmarks x (x, y, z) — fixed by MediaPipe
MIN_SAMPLES_WARN = 10
RANDOM_SEED = 42


def resample_to_length(frames, target_len):
    """Linearly resample a variable-length list of 126-dim frames to a
    fixed length. Handles recordings shorter or longer than target_len
    (camera frame rate isn't perfectly constant)."""
    frames = np.asarray(frames, dtype=np.float32)
    n = len(frames)
    if n == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)
    if n == target_len:
        return frames
    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)
    out = np.empty((target_len, FEATURE_DIM), dtype=np.float32)
    for d in range(FEATURE_DIM):
        out[:, d] = np.interp(new_idx, old_idx, frames[:, d])
    return out


def load_dataset(path):
    if not path.exists():
        print(f"ERROR: {path} not found. Move isl_dataset.json into this folder first.")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    if not raw:
        print("ERROR: dataset is empty.")
        sys.exit(1)
    return raw


def main():
    dataset_path = Path(__file__).parent / "isl_dataset.json"
    raw = load_dataset(dataset_path)

    labels_sorted = sorted({sample["label"] for sample in raw})
    label_to_idx = {label: i for i, label in enumerate(labels_sorted)}

    print(f"Found {len(labels_sorted)} unique labels, {len(raw)} total samples.\n")
    print("Samples per label:")
    counts = {label: 0 for label in labels_sorted}
    for sample in raw:
        counts[sample["label"]] += 1
    for label in labels_sorted:
        flag = "  <-- WARNING: fewer than 10, recognition for this sign will be weak" \
            if counts[label] < MIN_SAMPLES_WARN else ""
        print(f"  {label:10s} {counts[label]}{flag}")
    print()

    X = np.stack([resample_to_length(s["frames"], SEQUENCE_LEN) for s in raw])
    y = np.array([label_to_idx[s["label"]] for s in raw])

    # Stratified split needs at least 2 samples per class; fall back to a
    # plain random split if any label is too sparse for stratification.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
    except ValueError:
        print("Note: some labels have too few samples to stratify the split; using a plain random split.\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

    num_classes = len(labels_sorted)

    model = models.Sequential([
        layers.Input(shape=(SEQUENCE_LEN, FEATURE_DIM)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=16,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc * 100:.1f}%")
    if test_acc < 0.8:
        print("Below 80% — record more samples (especially for labels flagged above) and rerun.")

    model.save("isl_model.h5")
    with open("labels.json", "w") as f:
        json.dump(labels_sorted, f)

    print("\nSaved isl_model.h5 and labels.json.")
    print("Next: python convert_model.py")
    print("      tensorflowjs_converter --input_format keras isl_model_clean.keras ../model")


if __name__ == "__main__":
    main()
