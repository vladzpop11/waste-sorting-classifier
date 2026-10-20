"""One-time evaluation of the exported model on the test split.

Refuses to run again once reports/test_metrics.json exists (FORCE=1 overrides): using the
test set twice would turn it into a validation set. Scores the ONNX file that the API serves
(its SHA-256 is recorded), with 95% bootstrap intervals over test photos.
"""

import os
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.artifacts import read_json, write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.split import load_split
from waste_sorting_classifier.train import save_confusion

N_BOOTSTRAP = 1000


def bootstrap(y: np.ndarray, pred: np.ndarray, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        acc.append(accuracy_score(y[i], pred[i]))
        f1.append(f1_score(y[i], pred[i], average="macro", zero_division=0))
    return {
        "accuracy": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
        "macro_f1": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
    }


def main() -> None:
    cfg = load_config()
    reports = project_path(cfg["paths"]["reports_dir"])
    out = reports / "test_metrics.json"
    if out.exists() and os.environ.get("FORCE") != "1":
        sys.exit(f"{out} already exists: the test set has been used. Set FORCE=1 to override.")

    model = OnnxClassifier(project_path(cfg["paths"]["models_dir"]))
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    test = load_split(cfg).query("split == 'test'").sort_values("path")
    batch = np.stack([pp.preprocess(pp.load_rgb(images_dir / p)) for p in test["path"]])
    proba = np.concatenate(
        [model.predict_proba(batch[i : i + 64]) for i in range(0, len(batch), 64)]
    )

    classes = model.classes
    y = np.array([classes.index(label) for label in test["label"]])
    pred = proba.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, zero_division=0)
    selection = read_json(reports / "model_selection.json") or {}
    record = {
        "images": len(y),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "ci95": bootstrap(y, pred, cfg["seed"]),
        "per_class": {
            c: {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }  # fmt: skip
            for k, c in enumerate(classes)
        },
        "confusion": confusion_matrix(y, pred).tolist(),
        "classes": classes,
        "arch": model.meta["arch"],
        "onnx_sha256": model.meta["onnx_sha256"],
        "checkpoint_sha256": model.meta["checkpoint_sha256"],
        "validation": selection.get("runs", {}).get(model.meta["arch"], {}),
    }
    write_json(out, record)
    save_confusion(
        np.array(record["confusion"]),
        classes,
        project_path(cfg["paths"]["figures_dir"]) / "confusion_test.png",
    )
    ci = record["ci95"]
    for name in ["accuracy", "macro_f1"]:
        low, high = ci[name]
        print(f"test {name}: {record[name]:.4f} (95% CI {low:.4f}-{high:.4f})")
    for c, m in record["per_class"].items():
        scores = " ".join(f"{k} {m[k]:.3f}" for k in ["precision", "recall", "f1"])
        print(f"  {c:10s} {scores} (n={m['support']})")


if __name__ == "__main__":
    main()
