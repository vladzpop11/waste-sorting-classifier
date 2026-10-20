"""Build and measure the familiarity check (familiarity.py): bank, threshold, report.

The classifier always answers with one of its six classes, even for a photo of a person or
a landscape. This check compares the model's features for a photo (the 1280 numbers its
final layer reads) with those of every training photo, by cosine similarity, and calls the
photo unfamiliar when even the nearest training photo is not similar enough.

- The training features are stored as unit vectors rounded to 8 bits (models/familiarity.bin,
  2.3 MB), the same file the API and the browser demo use.
- Threshold, by the rule in configs/config.yaml: the value that lets 95% of validation photos
  through. Its false-alarm rate on new waste photos is estimated by setting it on one half
  of the validation groups and counting on the other half, both ways.
- Everyday photos (Imagenette) measure how many unfamiliar photos it catches; the photos
  used to choose this method are excluded from that measurement.
The test split is not used. Writes models/familiarity.bin, the "familiarity" record in
models/model.json, reports/familiarity.json and reports/figures/familiarity.png.

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
from waste_sorting_classifier.familiarity import BANK_FILE, SCALE, quantize, similarity
from waste_sorting_classifier.fetch_model import fetch
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.split import load_split

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
    bins = np.linspace(min(val.min(), everyday.min()), 1, 50)
    ax.hist(everyday, bins, alpha=0.6, color="#b4532a", label=f"everyday photos ({len(everyday)})",
            density=True)  # fmt: skip
    ax.hist(val, bins, alpha=0.6, color="#2f7d5b", label=f"validation waste photos ({len(val)})",
            density=True)  # fmt: skip
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.3f}")
    ax.set(xlabel="similarity to the nearest training photo", ylabel="density",
           title="Familiarity check")  # fmt: skip
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

    bank = quantize(features_of(model, [images / p for p in train["path"]])[1])
    val_logits, val_features = features_of(model, [images / p for p in val["path"]])
    val_sim = similarity(bank, val_features)
    correct = val_logits.argmax(1) == np.array([model.classes.index(c) for c in val["label"]])
    keep = cfg["familiarity"]["keep"]
    threshold = float(np.quantile(val_sim, 1 - keep))

    # False alarms on new photos: threshold from one half of the validation groups, counted
    # on the other half, both ways.
    groups = sorted(val["group"].unique())
    random.Random(cfg["seed"]).shuffle(groups)
    half = val["group"].isin(groups[: len(groups) // 2]).to_numpy()
    flagged = []
    for fit, held in [(half, ~half), (~half, half)]:
        t = np.quantile(val_sim[fit], 1 - keep)
        flagged.append(float((val_sim[held] < t).mean()))

    chosen, held_out = ood_photos(cfg)
    everyday_logits, everyday_features = features_of(model, held_out)
    everyday_sim = similarity(bank, everyday_features)
    caught = everyday_sim < threshold
    by_class = {}
    for path, c in zip(held_out, caught, strict=True):
        by_class.setdefault(IMAGENETTE[path.parent.name], []).append(bool(c))
    proba = np.exp(everyday_logits - everyday_logits.max(1, keepdims=True))
    proba /= proba.sum(1, keepdims=True)

    bank_path = models / BANK_FILE
    bank.tofile(bank_path)
    meta = json.loads((models / "model.json").read_text())
    meta["familiarity"] = {
        "file": BANK_FILE,
        "sha256": sha256(bank_path),
        "shape": list(bank.shape),
        "scale": SCALE,
        "threshold": threshold,
        "rule": f"lets {keep:.0%} of validation photos through",
    }
    write_json(models / "model.json", meta)
    report = {
        "onnx_sha256": meta["onnx_sha256"],
        "threshold": threshold,
        "rule": meta["familiarity"]["rule"],
        "bank_photos": int(bank.shape[0]),
        "validation_photos": len(val_sim),
        "unfamiliar_validation_photos": sorted(val["path"][val_sim < threshold]),
        # Descriptive (the threshold was set on these photos): flagged photos are also the
        # ones the model often gets wrong.
        "validation_accuracy": {
            "flagged": float(correct[val_sim < threshold].mean()),
            "not_flagged": float(correct[val_sim >= threshold].mean()),
        },
        "false_alarm_rate_on_new_waste_photos": {
            "halves": flagged,
            "mean": float(np.mean(flagged)),
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
    save_figure(val_sim, everyday_sim, threshold,
                project_path(cfg["paths"]["figures_dir"]) / "familiarity.png")  # fmt: skip
    e = report["everyday_photos"]
    print(f"threshold {threshold:.4f} ({meta['familiarity']['rule']}); false alarms on new "
          f"waste photos {np.mean(flagged):.1%}; everyday photos caught {e['caught']:.1%} "
          f"of {e['photos']}")  # fmt: skip


if __name__ == "__main__":
    main()
