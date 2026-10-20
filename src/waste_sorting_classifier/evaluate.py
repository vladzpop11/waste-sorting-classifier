"""One-time evaluation of the exported model on the test split.

Refuses to run again once reports/test_metrics.json exists (FORCE=1 overrides): using the
test set twice would turn it into a validation set. Scores the ONNX file that the API serves
(its SHA-256 is recorded) and checks that it was exported from the checkpoint the selection
rule chose. Reports:
- accuracy, macro F1, per-class scores, with 95% intervals from a bootstrap over groups;
- calibration (ECE and a reliability diagram);
- accuracy by how similar each test photo is to its nearest training photo of the same
  class, to show how much the score depends on near-duplicates of training photos;
- every test prediction, so later analyses never need to rerun the test set.

python -m waste_sorting_classifier.evaluate --rehearse runs the same code on the validation
split and writes to .cache/rehearsal/, to check the script before its single real run.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.artifacts import read_json, write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.metrics import grouped_bootstrap, reliability
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.plots import save_confusion, save_reliability
from waste_sorting_classifier.split import load_split

SIMILARITY_BANDS = [0.0, 0.75, 0.80, 0.85]  # then up to the split threshold (0.86)


def nearest_train_similarity(cfg: dict, split: pd.DataFrame, part: str) -> pd.Series:
    """For each photo of `part`: the highest same-class cosine similarity to a training photo,
    from the embeddings `make split` saved (rows in data/processed/images.csv order)."""
    processed = project_path(cfg["data"]["processed_dir"])
    index = pd.read_csv(processed / "images.csv")
    embeddings = np.load(processed / "embeddings.npy")
    if len(embeddings) != len(index) or not index["path"].equals(split["path"]):
        raise SystemExit("embeddings.npy does not match the split: run `make split` first")
    labels, parts = split["label"].to_numpy(), split["split"].to_numpy()
    test, train = np.flatnonzero(parts == part), np.flatnonzero(parts == "train")
    sims = embeddings[test] @ embeddings[train].T
    same = labels[test][:, None] == labels[train][None, :]
    return pd.Series(np.where(same, sims, -1).max(1), index=split["path"].iloc[test])


def main() -> None:
    cfg = load_config()
    reports = project_path(cfg["paths"]["reports_dir"])
    rehearse = "--rehearse" in sys.argv[1:]
    part = "val" if rehearse else "test"
    figures = (
        project_path(".cache/rehearsal") if rehearse else project_path(cfg["paths"]["figures_dir"])
    )
    out = figures / "val_metrics.json" if rehearse else reports / "test_metrics.json"
    figures.mkdir(parents=True, exist_ok=True)
    if not rehearse and out.exists() and os.environ.get("FORCE") != "1":
        sys.exit(f"{out} already exists: the test set has been used. Set FORCE=1 to override.")

    # Every precondition is checked before the test photos are touched.
    model = OnnxClassifier(project_path(cfg["paths"]["models_dir"]))
    selection = read_json(reports / "model_selection.json")
    if not selection and rehearse:  # rehearsing before any selection exists
        print("warning: no reports/model_selection.json; rehearsing without the selection check")
        selection = {"chosen_checkpoint_sha256": model.meta["checkpoint_sha256"],
                     "chosen_run": None, "runs": {None: None}}  # fmt: skip
    if not selection:
        sys.exit("reports/model_selection.json is missing: run `make select` and `make export`")
    if model.meta["checkpoint_sha256"] != selection["chosen_checkpoint_sha256"]:
        sys.exit("models/model.onnx was not exported from the selected checkpoint")
    split = load_split(cfg)
    similarity = nearest_train_similarity(cfg, split, part)

    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    test = split[split["split"] == part].sort_values("path").reset_index(drop=True)
    proba = np.concatenate([
        model.predict_proba(np.stack([pp.preprocess(pp.load_rgb(images_dir / p))
                                      for p in test["path"].iloc[i : i + 64]]))
        for i in range(0, len(test), 64)
    ])  # fmt: skip

    classes = model.classes
    y = np.array([classes.index(label) for label in test["label"]])
    pred = proba.argmax(1)
    correct = pred == y
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, zero_division=0)
    ci = grouped_bootstrap(y, pred, test["group"].to_numpy(), len(classes), cfg["seed"])
    calibration = reliability(proba, y)
    test["similarity"] = similarity.loc[test["path"]].to_numpy()
    edges = [*SIMILARITY_BANDS, cfg["split"]["similarity_threshold"]]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (test["similarity"] >= lo).to_numpy() & (test["similarity"] < hi).to_numpy()
        if inside.any():
            bands.append({"from": lo, "to": hi, "photos": int(inside.sum()),
                          "accuracy": float(correct[inside].mean())})  # fmt: skip

    record = {
        "images": len(y),
        "groups": int(test["group"].nunique()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "ci95": ci,
        "per_class": {
            c: {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
                "f1_ci95": ci["per_class_f1"][k],
                "recall_ci95": ci["per_class_recall"][k],
            }  # fmt: skip
            for k, c in enumerate(classes)
        },
        "confusion": confusion_matrix(y, pred).tolist(),
        "calibration": calibration,
        "accuracy_by_similarity_to_train": bands,
        "classes": classes,
        "arch": model.meta["arch"],
        "onnx_sha256": model.meta["onnx_sha256"],
        "checkpoint_sha256": model.meta["checkpoint_sha256"],
        "selected_run": selection["chosen_run"],
        "validation": selection["runs"][selection["chosen_run"]],
        "predictions": [
            {
                "path": r.path,
                "label": r.label,
                "predicted": classes[p],
                "confidence": round(float(proba[i].max()), 4),
                "similarity_to_train": round(float(r.similarity), 4),
            }  # fmt: skip
            for i, (r, p) in enumerate(zip(test.itertuples(), pred, strict=True))
        ],
    }
    write_json(out, record)
    save_confusion(np.array(record["confusion"]), classes, figures / f"confusion_{part}.png",
                   title=f"{part.capitalize()} confusion matrix")  # fmt: skip
    save_reliability(
        calibration, figures / f"reliability_{part}.png", f"{part.capitalize()} calibration"
    )

    for name in ["accuracy", "macro_f1"]:
        low, high = ci[name]
        print(f"{part} {name}: {record[name]:.4f} (95% CI {low:.4f}-{high:.4f}, grouped)")
    for c, m in record["per_class"].items():
        scores = " ".join(f"{k} {m[k]:.3f}" for k in ["precision", "recall", "f1"])
        print(f"  {c:10s} {scores} (n={m['support']})")
    print(f"ECE {calibration['ece']:.4f}")
    for b in bands:
        print(f"  similarity {b['from']:.2f}-{b['to']:.2f}: {b['accuracy']:.3f} ({b['photos']})")


if __name__ == "__main__":
    main()
