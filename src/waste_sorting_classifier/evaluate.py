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
--recompute-intervals recomputes the intervals from the recorded test predictions, without
running the model on the test photos again (used once, to fix a bootstrap bug; see
"revisions" in reports/test_metrics.json).
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
from waste_sorting_classifier.metrics import bootstrap_intervals, reliability
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


def add_intervals(record: dict, y: np.ndarray, pred: np.ndarray, groups: np.ndarray,
                  seed: int) -> None:  # fmt: skip
    ci = bootstrap_intervals(y, pred, groups, len(record["classes"]), seed)
    record["ci95"] = {way: {k: ci[way][k] for k in ["resamples", "units", "accuracy", "macro_f1"]}
                      for way in ci}  # fmt: skip
    for k, c in enumerate(record["classes"]):
        for score in ["f1", "recall"]:
            record["per_class"][c][f"{score}_ci95"] = {
                way: ci[way][f"per_class_{score}"][k] for way in ci
            }


def report(record: dict, part: str) -> None:
    for name in ["accuracy", "macro_f1"]:
        ways = ", ".join(f"{way} {lo:.4f}-{hi:.4f}" for way, v in record["ci95"].items()
                         for lo, hi in [v[name]])  # fmt: skip
        print(f"{part} {name}: {record[name]:.4f} (95% CI {ways})")
    for c, m in record["per_class"].items():
        scores = " ".join(f"{k} {m[k]:.3f}" for k in ["precision", "recall", "f1"])
        lo, hi = m["f1_ci95"]["per_photo"]
        print(f"  {c:10s} {scores} (n={m['support']}, F1 per-photo CI {lo:.3f}-{hi:.3f})")
    print(f"ECE {record['calibration']['ece']:.4f}")
    for b in record["accuracy_by_similarity_to_train"]:
        print(f"  similarity {b['from']:.2f}-{b['to']:.2f}: {b['accuracy']:.3f} ({b['photos']})")


def recompute_intervals(cfg: dict, path) -> None:
    record = read_json(path)
    groups = load_split(cfg, verify=False).set_index("path")["group"]
    paths = [r["path"] for r in record["predictions"]]
    y = np.array([record["classes"].index(r["label"]) for r in record["predictions"]])
    pred = np.array([record["classes"].index(r["predicted"]) for r in record["predictions"]])
    add_intervals(record, y, pred, groups.loc[paths].to_numpy(), cfg["seed"])
    record.setdefault("revisions", []).append(
        "Intervals recomputed from the recorded predictions (the model was not run again): "
        "the first run's grouped bootstrap scored a class missing from a resample as F1 0, "
        "and per-photo intervals were added."
    )
    write_json(path, record)
    report(record, "test")


def main() -> None:
    cfg = load_config()
    reports = project_path(cfg["paths"]["reports_dir"])
    if "--recompute-intervals" in sys.argv[1:]:
        return recompute_intervals(cfg, reports / "test_metrics.json")
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
    add_intervals(record, y, pred, test["group"].to_numpy(), cfg["seed"])
    write_json(out, record)
    save_confusion(np.array(record["confusion"]), classes, figures / f"confusion_{part}.png",
                   title=f"{part.capitalize()} confusion matrix")  # fmt: skip
    save_reliability(
        calibration, figures / f"reliability_{part}.png", f"{part.capitalize()} calibration"
    )

    report(record, part)


if __name__ == "__main__":
    main()
