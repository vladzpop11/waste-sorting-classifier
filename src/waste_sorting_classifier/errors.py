"""Validation error analysis of the exported model: the photos it gets wrong, its
calibration, and the confidence below which the demo flags an answer as uncertain.

Uses the validation split only (the test set stays reserved). Writes
reports/val_errors.json and reports/figures/{val_errors,reliability_val}.png.
"""

import numpy as np

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.metrics import low_confidence_threshold, reliability
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.plots import _pyplot, save_reliability
from waste_sorting_classifier.split import load_split

BATCH = 64


def main() -> None:
    cfg = load_config()
    model = OnnxClassifier(project_path(cfg["paths"]["models_dir"]))
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    figures = project_path(cfg["paths"]["figures_dir"])
    val = load_split(cfg).query("split == 'val'").sort_values("path").reset_index(drop=True)
    proba = np.concatenate([
        model.predict_proba(np.stack([pp.preprocess(pp.load_rgb(images_dir / p))
                                      for p in val["path"].iloc[i : i + BATCH]]))
        for i in range(0, len(val), BATCH)
    ])  # fmt: skip
    y = np.array([model.classes.index(label) for label in val["label"]])
    val["predicted"] = [model.classes[k] for k in proba.argmax(1)]
    val["confidence"] = proba.max(1)
    correct = (val["predicted"] == val["label"]).to_numpy()
    wrong = val[~correct].sort_values("confidence", ascending=False)
    calibration = reliability(proba, y)
    flag = low_confidence_threshold(val["confidence"].to_numpy(), correct)

    plt = _pyplot()
    cols = 5
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), wrong.itertuples(), strict=False):
        ax.imshow(pp.load_rgb(images_dir / r.path))
        ax.set_title(f"{r.label} → {r.predicted}\n({r.confidence:.0%})", fontsize=9)
    fig.suptitle(f"Validation errors: {len(wrong)} of {len(val)} (true → predicted, confidence)")
    fig.tight_layout()
    fig.savefig(figures / "val_errors.png", dpi=110)
    plt.close(fig)
    save_reliability(calibration, figures / "reliability_val.png", "Validation calibration")

    pairs = wrong.groupby(["label", "predicted"]).size().sort_values(ascending=False)
    write_json(
        project_path(cfg["paths"]["reports_dir"]) / "val_errors.json",
        {
            "onnx_sha256": model.meta["onnx_sha256"],
            "errors": len(wrong),
            "images": len(val),
            "accuracy": float(correct.mean()),
            "calibration": calibration,
            "low_confidence": flag,
            "pairs": {f"{t} -> {p}": int(n) for (t, p), n in pairs.items()},
            "photos": [
                {
                    "path": r.path,
                    "label": r.label,
                    "predicted": r.predicted,
                    "confidence": round(float(r.confidence), 4),
                }  # fmt: skip
                for r in wrong.itertuples()
            ],
        },
    )
    print(f"{len(wrong)} of {len(val)} validation photos wrong; ECE {calibration['ece']:.4f}")
    if flag["threshold"] is None:
        print("no confidence threshold reaches the target accuracy: no low-confidence note")
    else:
        below, above = flag["accuracy_below"], flag["accuracy_above"]
        print(f"flag below {flag['threshold']:.2f} confidence: {flag['share_below']:.1%} of "
              f"photos, accuracy {below} below and {above} above")  # fmt: skip


if __name__ == "__main__":
    main()
