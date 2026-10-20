"""Familiarity check at serving time: is a photo like the training photos at all?

The classifier always answers with one of its six classes, even for a photo of a person or
a landscape. This compares the model's features for a photo (the 1280 numbers its final
layer reads) with those of every training photo, by cosine similarity: the photo is
unfamiliar when even the nearest training photo is not similar enough. The bank and the
threshold are built by familiarity_build.py. Light on purpose: the API image has NumPy and
ONNX Runtime only.
"""

from pathlib import Path

import numpy as np

from waste_sorting_classifier.artifacts import sha256

SCALE = 127  # unit vectors stored as round(127 * value), int8
BANK_FILE = "familiarity.bin"


def quantize(features: np.ndarray) -> np.ndarray:
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    return np.round(unit * SCALE).astype(np.int8)


def similarity(bank: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Cosine similarity of each photo to its nearest bank photo, in float64 (the browser
    demo computes the same in web/familiarity.js: unit query vector against the 8-bit bank,
    divided by 127)."""
    unit = features.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    return (unit @ bank.astype(np.float64, copy=False).T).max(axis=1) / SCALE


class Familiarity:
    """The stored bank and threshold, as the API uses them."""

    def __init__(self, models_dir: Path, meta: dict):
        record = meta["familiarity"]
        path = models_dir / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"{path} does not match the SHA-256 recorded in model.json")
        # Kept as float64 once, rather than converted on every request.
        self.bank = np.fromfile(path, dtype=np.int8).reshape(record["shape"]).astype(np.float64)
        self.threshold = record["threshold"]

    def score(self, features: np.ndarray) -> np.ndarray:
        return similarity(self.bank, features)
