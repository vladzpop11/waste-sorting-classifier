"""Temperature scaling, fitted on validation photos, so that the displayed confidence means
what it says. Labels never change.

The model is underconfident: label smoothing in training keeps its probabilities away from
0 and 1. Dividing the logits by one number, the temperature, before the softmax fixes most
of that (a temperature below 1 sharpens the probabilities).
- Rule (configs/config.yaml, calibration): the temperature minimises the negative
  log-likelihood of the validation photos.
- Its effect on new photos is measured by fitting on one half of the validation groups and
  scoring the other half, both ways (the halves of the familiarity check).
- The test split is not used: its one evaluation recorded uncalibrated probabilities.
Writes the "calibration" record in models/model.json and reports/calibration.json.

python -m waste_sorting_classifier.calibrate
"""

import json
import random

import numpy as np
from scipy.optimize import minimize_scalar

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.metrics import reliability
from waste_sorting_classifier.onnx_model import OnnxClassifier, softmax
from waste_sorting_classifier.split import load_split

BATCH = 64


def nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood of the true classes, computed stably."""
    z = logits.astype(np.float64) / temperature
    z -= z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    found = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20), method="bounded",
                            options={"xatol": 1e-6})  # fmt: skip
    return float(found.x)


def scores(logits: np.ndarray, y: np.ndarray, temperature: float) -> dict:
    proba = softmax(logits / temperature)
    return {
        "photos": len(y),
        "accuracy": float((proba.argmax(1) == y).mean()),
        "mean_confidence": float(proba.max(1).mean()),
        "ece": reliability(proba, y)["ece"],
        "nll": nll(logits, y, temperature),
    }


def main() -> None:
    cfg = load_config()
    models = project_path(cfg["paths"]["models_dir"])
    model = OnnxClassifier(models)
    images = project_path(cfg["data"]["raw_dir"]) / "images"
    val = load_split(cfg).query("split == 'val'").sort_values("path").reset_index(drop=True)
    logits = np.concatenate([
        model.run(np.stack([pp.preprocess(pp.load_rgb(images / p))
                            for p in val["path"].iloc[i : i + BATCH]]))[0]
        for i in range(0, len(val), BATCH)
    ])  # fmt: skip
    y = np.array([model.classes.index(label) for label in val["label"]])
    temperature = fit_temperature(logits, y)

    # On new photos: fit on one half of the validation groups, score the other half.
    groups = sorted(val["group"].unique())
    random.Random(cfg["seed"]).shuffle(groups)
    half = val["group"].isin(groups[: len(groups) // 2]).to_numpy()
    held_out = []
    for fit, held in [(half, ~half), (~half, half)]:
        t = fit_temperature(logits[fit], y[fit])
        by_source = {}
        for source in sorted(val["source"].unique()):
            m = held & (val["source"] == source).to_numpy()
            by_source[source] = {"before": scores(logits[m], y[m], 1.0),
                                 "after": scores(logits[m], y[m], t)}  # fmt: skip
        held_out.append({
            "temperature": t,
            "before": scores(logits[held], y[held], 1.0),
            "after": scores(logits[held], y[held], t),
            "by_source": by_source,
        })  # fmt: skip

    meta = json.loads((models / "model.json").read_text())
    meta["calibration"] = {
        "method": "temperature",
        "temperature": temperature,
        "rule": "minimises the negative log-likelihood of the validation photos",
        "validation_photos": len(y),
    }
    write_json(models / "model.json", meta)
    mean = lambda key, when: float(np.mean([h[when][key] for h in held_out]))  # noqa: E731
    report = {
        "onnx_sha256": meta["onnx_sha256"],
        **meta["calibration"],
        # In-sample: the temperature was fitted on these photos.
        "validation": {"before": scores(logits, y, 1.0), "after": scores(logits, y, temperature)},
        "held_out_halves": held_out,
        "held_out_mean": {
            when: {k: mean(k, when) for k in ["accuracy", "mean_confidence", "ece", "nll"]}
            for when in ["before", "after"]
        },
    }
    write_json(project_path(cfg["paths"]["reports_dir"]) / "calibration.json", report)
    h = report["held_out_mean"]
    print(f"temperature {temperature:.4f}; on held-out validation halves: ECE "
          f"{h['before']['ece']:.4f} -> {h['after']['ece']:.4f}, mean confidence "
          f"{h['before']['mean_confidence']:.3f} -> {h['after']['mean_confidence']:.3f} "
          f"(accuracy {h['before']['accuracy']:.3f})")  # fmt: skip


if __name__ == "__main__":
    main()
