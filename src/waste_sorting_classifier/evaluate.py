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

import json
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
from waste_sorting_classifier.artifacts import read_json, sha256, write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.familiarity import Familiarity
from waste_sorting_classifier.fetch_model import fetch
from waste_sorting_classifier.metrics import bootstrap_intervals, reliability
from waste_sorting_classifier.onnx_model import OnnxClassifier, softmax
from waste_sorting_classifier.plots import save_confusion, save_reliability
from waste_sorting_classifier.split import load_split, split_embeddings

SIMILARITY_BANDS = [0.0, 0.75, 0.80, 0.85]  # then up to the split threshold (0.86)


def nearest_train_similarity(cfg: dict, split: pd.DataFrame, part: str) -> pd.Series:
    """For each photo of `part`: the highest same-class cosine similarity to a training photo,
    from the embeddings `make split` saved."""
    embeddings = split_embeddings(cfg, split)
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


def run_model(model: OnnxClassifier, paths: list) -> tuple[np.ndarray, np.ndarray]:
    logits, features = [], []
    for i in range(0, len(paths), 64):
        lo, fe = model.run(np.stack([pp.preprocess(pp.load_rgb(p)) for p in paths[i : i + 64]]))
        logits.append(lo)
        features.append(fe)
    return np.concatenate(logits), np.concatenate(features)


def summary(rows: pd.DataFrame, proba: np.ndarray, classes: list[str], cfg: dict) -> dict:
    """Scores, intervals, calibration and accuracy by similarity to training, for `rows`
    (which carry label, group and similarity)."""
    y = np.array([classes.index(label) for label in rows["label"]])
    pred = proba.argmax(1)
    correct = pred == y
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(classes))), zero_division=0
    )
    edges = [*SIMILARITY_BANDS, cfg["split"]["similarity_threshold"]]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = ((rows["similarity"] >= lo) & (rows["similarity"] < hi)).to_numpy()
        if inside.any():
            bands.append({"from": lo, "to": hi, "photos": int(inside.sum()),
                          "accuracy": float(correct[inside].mean())})  # fmt: skip
    record = {
        "images": len(y),
        "groups": int(rows["group"].nunique()),
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
        "confusion": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist(),
        "calibration": reliability(proba, y),
        "accuracy_by_similarity_to_train": bands,
        "classes": classes,
    }
    add_intervals(record, y, pred, rows["group"].to_numpy(), cfg["seed"])
    return record


def paired_difference(a: np.ndarray, b: np.ndarray, groups: np.ndarray, seed: int) -> dict:
    """Accuracy of a minus accuracy of b on the same photos, with a 95% interval from
    resampling whole groups."""
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    samples = []
    for _ in range(2000):
        i = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        samples.append(a[i].mean() - b[i].mean())
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"difference": float(a.mean() - b.mean()), "ci95": [float(low), float(high)]}


def comparison_model(cfg: dict) -> tuple[OnnxClassifier, Familiarity]:
    """The earlier version's served files, downloaded and checked against its record."""
    comp = cfg["compare"]
    meta = json.loads(project_path(comp["model_json"]).read_text())
    folder = project_path(comp["dir"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.json").write_text(json.dumps(meta))
    files = {
        "model.onnx": meta["onnx_sha256"],
        meta["familiarity"]["file"]: meta["familiarity"]["sha256"],
    }
    for name, expected in files.items():
        if not ((folder / name).exists() and sha256(folder / name) == expected):
            fetch(comp["base_url"] + name, folder / name, expected)
    model = OnnxClassifier(folder)
    return model, Familiarity(folder, model.meta)


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
    models = project_path(cfg["paths"]["models_dir"])
    model = OnnxClassifier(models)
    familiarity = Familiarity(models, model.meta)
    selection = read_json(reports / "model_selection.json")
    if not selection:
        sys.exit("reports/model_selection.json is missing: run `make select` and `make export`")
    if model.meta["checkpoint_sha256"] != selection["chosen_checkpoint_sha256"]:
        sys.exit("models/model.onnx was not exported from the selected checkpoint")
    old_model, old_familiarity = comparison_model(cfg)
    split = load_split(cfg)
    similarity = nearest_train_similarity(cfg, split, part)

    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    test = split[split["split"] == part].sort_values("path").reset_index(drop=True)
    test["similarity"] = similarity.loc[test["path"]].to_numpy()
    paths = [images_dir / p for p in test["path"]]
    logits, features = run_model(model, paths)
    old_logits, old_features = run_model(old_model, paths)

    classes = model.classes
    proba, old_proba = softmax(logits), softmax(old_logits)
    familiar = familiarity.score(logits, features) >= familiarity.threshold
    old_familiar = old_familiarity.score(old_logits, old_features) >= old_familiarity.threshold
    y = np.array([classes.index(label) for label in test["label"]])
    correct, old_correct = proba.argmax(1) == y, old_proba.argmax(1) == y

    record = summary(test, proba, classes, cfg)
    record["by_source"] = {}
    record[f"compared_with_{cfg['compare']['name']}"] = {}
    for source in sorted(test["source"].unique()):
        m = (test["source"] == source).to_numpy()
        record["by_source"][source] = summary(test[m], proba[m], classes, cfg)
        record["by_source"][source]["flagged_unfamiliar"] = float((~familiar[m]).mean())
        old = summary(test[m], old_proba[m], classes, cfg)
        record[f"compared_with_{cfg['compare']['name']}"][source] = {
            "accuracy": old["accuracy"],
            "macro_f1": old["macro_f1"],
            "ci95": old["ci95"],
            "flagged_unfamiliar": float((~old_familiar[m]).mean()),
            "accuracy_difference": paired_difference(
                correct[m], old_correct[m], test["group"].to_numpy()[m], cfg["seed"]
            ),
        }
    record.update(
        {
            "arch": model.meta["arch"],
            "onnx_sha256": model.meta["onnx_sha256"],
            "checkpoint_sha256": model.meta["checkpoint_sha256"],
            "selected_run": selection["chosen_run"],
            "validation": selection["runs"][selection["chosen_run"]],
            "compared_model_onnx_sha256": old_model.meta["onnx_sha256"],
            "predictions": [
                {
                    "path": r.path,
                    "source": r.source,
                    "label": r.label,
                    "predicted": classes[int(proba[i].argmax())],
                    "confidence": round(float(proba[i].max()), 4),
                    "familiar": bool(familiar[i]),
                    "similarity_to_train": round(float(r.similarity), 4),
                    f"{cfg['compare']['name']}_predicted": classes[int(old_proba[i].argmax())],
                    f"{cfg['compare']['name']}_familiar": bool(old_familiar[i]),
                }  # fmt: skip
                for i, r in enumerate(test.itertuples())
            ],
        }
    )
    write_json(out, record)
    save_confusion(np.array(record["confusion"]), classes, figures / f"confusion_{part}.png",
                   title=f"{part.capitalize()} confusion matrix")  # fmt: skip
    save_reliability(record["calibration"], figures / f"reliability_{part}.png",
                     f"{part.capitalize()} calibration")  # fmt: skip
    report(record, part)
    for source, r in record["by_source"].items():
        old = record[f"compared_with_{cfg['compare']['name']}"][source]
        d = old["accuracy_difference"]
        print(f"{source:10s} accuracy {r['accuracy']:.4f} (v3 {old['accuracy']:.4f}; difference "
              f"{d['difference']:+.4f}, 95% CI {d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f}) | "
              f"macro F1 {r['macro_f1']:.4f} (v3 {old['macro_f1']:.4f}) | flagged unfamiliar "
              f"{r['flagged_unfamiliar']:.1%} (v3 {old['flagged_unfamiliar']:.1%})")  # fmt: skip


if __name__ == "__main__":
    main()
