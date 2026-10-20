"""The exported model as used by evaluation, the API and the demo: ONNX Runtime only."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from waste_sorting_classifier.artifacts import sha256


class OnnxClassifier:
    """models/model.onnx + models/model.json; refuses a model file that differs from its record."""

    def __init__(self, models_dir: Path, threads: int = 1):
        self.meta = json.loads((models_dir / "model.json").read_text())
        path = models_dir / "model.onnx"
        if sha256(path) != self.meta["onnx_sha256"]:
            raise ValueError(f"{path} does not match the SHA-256 recorded in model.json")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.classes: list[str] = self.meta["classes"]

    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        """(N, 3, 224, 224) preprocessed images -> (N, classes) probabilities."""
        logits = self.session.run(["logits"], {"image": batch.astype(np.float32)})[0]
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
