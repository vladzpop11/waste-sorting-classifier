"""Build and measure the familiarity check (familiarity.py): threshold and report.

The classifier always answers with one of its six classes, even for a photo of a person or
a landscape. The check scores how familiar a photo looks to the model and flags it below a
threshold. Version 4 uses the Mahalanobis distance of the model's features to the training
class means (shared covariance, regularised by 0.001 on the diagonal). Chosen on the
Imagenette photos set aside for that, at 95% of validation photos kept: it caught 83.5% of
everyday photos and let 8.7% through with 70%+ confidence, against 82.5% / 15.8% for the
energy score and 73.0% / 17.0% for version 3's nearest-training-photo similarity (README;
confidence before calibration, which came later).

- Threshold, by the rule in configs/config.yaml: the value that lets 95% of validation photos
  through. Its false-alarm rate on new waste photos is estimated by setting it on one half
  of the validation groups and counting on the other half, both ways, per source too.
- Everyday photos (Imagenette) measure how many unfamiliar photos it catches; the photos
  used to choose the method are excluded from that measurement.
The test split is not used. Writes the "familiarity" record in models/model.json,
reports/familiarity.json and reports/figures/familiarity.png.

python -m waste_sorting_classifier.familiarity_build
"""

import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.artifacts import sha256, write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.familiarity import BANK_FILE, Familiarity
from waste_sorting_classifier.fetch_model import fetch
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.split import load_split

REGULARISATION = 1e-3  # added to the covariance's diagonal
IMAGENETTE = {
    "n01440764": "tench (fish, often held by people)",
    "n02102040": "English springer (dog)",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def download_ood(cfg: dict) -> Path:
    """Imagenette, checked against its SHA-256 and unpacked with tarfile's safe filter."""
    fam = cfg["familiarity"]
    root = project_path(fam["ood_dir"])
    archive = root / Path(fam["ood_url"]).name
    if not (archive.exists() and sha256(archive) == fam["ood_sha256"]):
        fetch(fam["ood_url"], archive, fam["ood_sha256"], timeout=600)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    return root


def ood_photos(cfg: dict) -> tuple[list[Path], list[Path]]:
    """Imagenette validation photos in the fixed order: (used to choose the method, held out)."""
    root = project_path(cfg["familiarity"]["ood_dir"]) / "imagenette2-160" / "val"
    if not root.exists():
        sys.exit(f"{root} is missing: run make familiarity-data")
    photos = sorted(root.rglob("*.JPEG"))
    random.Random(0).shuffle(photos)
    n = cfg["familiarity"]["ood_used_for_choosing"]
    return photos[:n], photos[n:]


def features_of(model: OnnxClassifier, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    logits, features = [], []
    for i in range(0, len(paths), 64):
        lo, fe = model.run(np.stack([pp.preprocess(pp.load_rgb(p)) for p in paths[i : i + 64]]))
        logits.append(lo)
        features.append(fe)
    return np.concatenate(logits), np.concatenate(features)


def save_figure(val: np.ndarray, everyday: np.ndarray, threshold: float, path: Path) -> None:
    from waste_sorting_classifier.plots import _pyplot

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(min(val.min(), everyday.min()), max(val.max(), everyday.max()), 50)
    ax.hist(everyday, bins, alpha=0.6, color="#b4532a", label=f"everyday photos ({len(everyday)})",
            density=True)  # fmt: skip
    ax.hist(val, bins, alpha=0.6, color="#2f7d5b", label=f"validation waste photos ({len(val)})",
            density=True)  # fmt: skip
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.3f}")
    ax.set(xlabel="familiarity score (minus the Mahalanobis distance to the nearest class)",
           ylabel="density", title="Familiarity check")  # fmt: skip
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    models = project_path(cfg["paths"]["models_dir"])
    model = OnnxClassifier(models)
    images = project_path(cfg["data"]["raw_dir"]) / "images"
    split = load_split(cfg)
    train = split[split["split"] == "train"].sort_values("path")
    val = split[split["split"] == "val"].sort_values("path").reset_index(drop=True)

    # Class means and shared inverse covariance of the training features (true labels),
    # stored as float32; everything below scores with the stored values, as the API does.
    _, train_features = features_of(model, [images / p for p in train["path"]])
    y = np.array([model.classes.index(c) for c in train["label"]])
    means = np.stack([train_features[y == k].mean(0) for k in range(len(model.classes))])
    covariance = np.cov((train_features - means[y]).T) + REGULARISATION * np.eye(means.shape[1])
    stored = np.concatenate([means.ravel(), np.linalg.inv(covariance).ravel()]).astype("<f4")
    stored.tofile(models / BANK_FILE)
    record = {"method": "mahalanobis", "file": BANK_FILE, "sha256": sha256(models / BANK_FILE),
              "classes": len(model.classes), "dims": int(means.shape[1]),
              "regularisation": REGULARISATION}  # fmt: skip
    meta = json.loads((models / "model.json").read_text())
    familiarity = Familiarity(models, {"familiarity": {**record, "threshold": 0.0}})

    val_logits, val_features = features_of(model, [images / p for p in val["path"]])
    val_score = familiarity.score(val_logits, val_features)
    correct = val_logits.argmax(1) == np.array([model.classes.index(c) for c in val["label"]])
    keep = cfg["familiarity"]["keep"]
    threshold = float(np.quantile(val_score, 1 - keep))

    # False alarms on new photos: threshold from one half of the validation groups, counted
    # on the other half, both ways.
    groups = sorted(val["group"].unique())
    random.Random(cfg["seed"]).shuffle(groups)
    half = val["group"].isin(groups[: len(groups) // 2]).to_numpy()
    flagged, by_source = [], {}
    for fit, held in [(half, ~half), (~half, half)]:
        t = np.quantile(val_score[fit], 1 - keep)
        flagged.append(float((val_score[held] < t).mean()))
        for source in sorted(val["source"].unique()):
            m = held & (val["source"] == source).to_numpy()
            by_source.setdefault(source, []).append(float((val_score[m] < t).mean()))

    _, held_out = ood_photos(cfg)
    everyday_logits, everyday_features = features_of(model, held_out)
    everyday_score = familiarity.score(everyday_logits, everyday_features)
    caught = everyday_score < threshold
    by_class = {}
    for path, c in zip(held_out, caught, strict=True):
        by_class.setdefault(IMAGENETTE[path.parent.name], []).append(bool(c))
    proba = model.probabilities(everyday_logits)  # calibrated, as the API shows them

    meta["familiarity"] = {**record, "threshold": threshold,
                           "rule": f"lets {keep:.0%} of validation photos through"}  # fmt: skip
    write_json(models / "model.json", meta)
    unfamiliar = val_score < threshold
    report = {
        "onnx_sha256": meta["onnx_sha256"],
        "method": "mahalanobis",
        "threshold": threshold,
        "rule": meta["familiarity"]["rule"],
        "validation_photos": len(val_score),
        "unfamiliar_validation_photos": sorted(val["path"][unfamiliar]),
        # Descriptive (the threshold was set on these photos): flagged photos are also the
        # ones the model often gets wrong.
        "validation_accuracy": {
            "flagged": float(correct[unfamiliar].mean()),
            "not_flagged": float(correct[~unfamiliar].mean()),
        },
        "false_alarm_rate_on_new_waste_photos": {
            "halves": flagged,
            "mean": float(np.mean(flagged)),
            "by_source": {k: float(np.mean(v)) for k, v in by_source.items()},
        },
        "everyday_photos": {
            "source": "Imagenette validation photos not used to choose the method",
            "photos": int(len(held_out)),
            "caught": float(caught.mean()),
            "caught_by_imagenette_class": {
                k: float(np.mean(v)) for k, v in sorted(by_class.items())
            },
            "model_confidence_median": float(np.median(proba.max(1))),
            "model_confidence_at_least_0.7": float((proba.max(1) >= 0.7).mean()),
            "missed_and_confidence_at_least_0.7": float((~caught & (proba.max(1) >= 0.7)).mean()),
        },
    }
    write_json(project_path(cfg["paths"]["reports_dir"]) / "familiarity.json", report)
    save_figure(val_score, everyday_score, threshold,
                project_path(cfg["paths"]["figures_dir"]) / "familiarity.png")  # fmt: skip
    e = report["everyday_photos"]
    print(f"threshold {threshold:.4f} ({meta['familiarity']['rule']}); false alarms on new "
          f"waste photos {np.mean(flagged):.1%}; everyday photos caught {e['caught']:.1%} "
          f"of {e['photos']}")  # fmt: skip


if __name__ == "__main__":
    main()
