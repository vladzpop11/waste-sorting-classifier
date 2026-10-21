"""Familiarity check at serving time: is a photo like the training photos at all?

The classifier always answers with one of its six classes, even for a photo of a person or
a landscape. The check gives each photo a familiarity score and calls it unfamiliar below
a threshold set on validation photos (familiarity_build.py). The method is recorded in
models/model.json:
- "mahalanobis" (version 4): minus the smallest Mahalanobis distance (squared) from the
  photo's features to a class mean of the training features, with their shared covariance
  (regularised). File: the class means, then the inverse covariance, float32.
- "energy": log-sum-exp of the model's six logits (compared when choosing; not deployed).
- "nearest" (version 3): cosine similarity of the model's features to the nearest training
  photo, against a stored bank of training features (8-bit unit vectors).
Light on purpose: the API image has NumPy and ONNX Runtime only.
"""

from pathlib import Path

import numpy as np

from waste_sorting_classifier.artifacts import sha256

SCALE = 127  # bank unit vectors stored as round(127 * value), int8
BANK_FILE = "familiarity.bin"


def energy(logits: np.ndarray) -> np.ndarray:
    """log(sum(exp(logits))), computed stably in float64 (web/familiarity.js matches it)."""
    z = logits.astype(np.float64)
    top = z.max(axis=1)
    return top + np.log(np.exp(z - top[:, np.newaxis]).sum(axis=1))


def mahalanobis(means: np.ndarray, precision: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Minus the smallest squared Mahalanobis distance to a class mean, in float64 (higher =
    more familiar; web/familiarity.js computes the same)."""
    d = features.astype(np.float64)[:, np.newaxis, :] - means[np.newaxis]  # (photos, classes, dims)
    return -((d @ precision) * d).sum(axis=2).min(axis=1)  # matrix products: ~2 ms per photo


def quantize(features: np.ndarray) -> np.ndarray:
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    return np.round(unit * SCALE).astype(np.int8)


def similarity(bank: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Cosine similarity of each photo to its nearest bank photo, in float64."""
    unit = features.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    return (unit @ bank.astype(np.float64, copy=False).T).max(axis=1) / SCALE


class Familiarity:
    """The recorded method and threshold, as the API uses them."""

    def __init__(self, models_dir: Path, meta: dict):
        record = meta["familiarity"]
        self.method = record.get("method", "nearest")
        self.threshold = record["threshold"]
        if self.method == "mahalanobis":
            path = models_dir / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"{path} does not match the SHA-256 recorded in model.json")
            k, n = record["classes"], record["dims"]
            values = np.fromfile(path, dtype="<f4").astype(np.float64)
            self.means = values[: k * n].reshape(k, n)
            self.precision = values[k * n :].reshape(n, n)
        elif self.method == "nearest":
            path = models_dir / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"{path} does not match the SHA-256 recorded in model.json")
            # Kept as float64 once, rather than converted on every request.
            self.bank = np.fromfile(path, dtype=np.int8).reshape(record["shape"]).astype(np.float64)

    def score(self, logits: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        if self.method == "mahalanobis":
            return mahalanobis(self.means, self.precision, features)
        if self.method == "energy":
            return energy(logits)
        return similarity(self.bank, features)
