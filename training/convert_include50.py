"""
Convert INCLUDE-50 MediaPipe Holistic .npy skeletons into isl_dataset.json
in the same 126-dim wrist-normalized format as data-collector/index.html.

Holistic layout per frame: 543 points (pose 33, face 468, left hand 21, right hand 21).
This app only uses the two hands. Left hand -> first 63 features, right hand -> last 63.
Keep extractFeatureVector() in the collector and recognizer on the same Left/Right slots.
"""

import json
from pathlib import Path

import numpy as np

POSE = 33
FACE = 468
HAND = 21
LEFT_START = POSE + FACE
RIGHT_START = LEFT_START + HAND
FEATURE_DIM = 126


def clean_label(folder_name):
    if "." in folder_name:
        return folder_name.split(".", 1)[1].strip()
    return folder_name.strip()


def normalize_hand(pts):
    """pts: (21, 3). Zeros if the hand was not detected."""
    if not np.isfinite(pts).all() or np.abs(pts).sum() < 1e-6:
        return np.zeros(HAND * 3, dtype=np.float32)
    wrist = pts[0]
    mcp = pts[9]
    scale = np.linalg.norm(mcp - wrist) or 1.0
    rel = (pts - wrist) / scale
    return rel.reshape(-1).astype(np.float32)


def frames_from_holistic(arr):
    """arr: (T, 543, 3) -> list of 126-float frames."""
    frames = []
    for t in range(arr.shape[0]):
        left = normalize_hand(arr[t, LEFT_START:RIGHT_START])
        right = normalize_hand(arr[t, RIGHT_START:RIGHT_START + HAND])
        if np.abs(left).sum() < 1e-6 and np.abs(right).sum() < 1e-6:
            continue
        vec = np.concatenate([left, right])
        frames.append([round(float(x), 5) for x in vec])
    return frames


def main():
    src = Path(__file__).parent / "include50_raw" / "skeletons_full"
    out = Path(__file__).parent / "isl_dataset.json"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Download INCLUDE-50 skeletons first.")

    dataset = []
    skipped = 0
    for npy in sorted(src.rglob("*.npy")):
        arr = np.load(npy)
        if arr.ndim != 3 or arr.shape[1:] != (543, 3):
            skipped += 1
            continue
        frames = frames_from_holistic(arr)
        if len(frames) < 5:
            skipped += 1
            continue
        dataset.append({"label": clean_label(npy.parent.name), "frames": frames})

    if not dataset:
        raise SystemExit("No usable samples.")

    with open(out, "w") as f:
        json.dump(dataset, f)

    labels = sorted({s["label"] for s in dataset})
    print(f"Wrote {out} with {len(dataset)} samples, {len(labels)} labels, skipped {skipped}.")
    from collections import Counter
    counts = Counter(s["label"] for s in dataset)
    for label in labels:
        print(f"  {label:20s} {counts[label]}")


if __name__ == "__main__":
    main()
