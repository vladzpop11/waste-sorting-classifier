"""Validation error gallery: the photos the exported model gets wrong, with their labels.

Uses the validation split only (the test set stays reserved). Writes
reports/figures/val_errors.png and reports/val_errors.json.
"""

import numpy as np

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.split import load_split


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    model = OnnxClassifier(project_path(cfg["paths"]["models_dir"]))
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    val = load_split(cfg).query("split == 'val'").sort_values("path").reset_index(drop=True)
    images = [pp.load_rgb(images_dir / p) for p in val["path"]]
    proba = model.predict_proba(np.stack([pp.preprocess(im) for im in images]))
    val["predicted"] = [model.classes[k] for k in proba.argmax(1)]
    val["confidence"] = proba.max(1)
    wrong = val[val["predicted"] != val["label"]].sort_values("confidence", ascending=False)

    cols = 5
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (i, r) in zip(np.atleast_1d(axes).ravel(), wrong.iterrows(), strict=False):
        ax.imshow(images[i])
        ax.set_title(f"{r['label']} → {r['predicted']}\n({r['confidence']:.0%})", fontsize=9)
    fig.suptitle(f"Validation errors: {len(wrong)} of {len(val)} (true → predicted, confidence)")
    fig.tight_layout()
    out = project_path(cfg["paths"]["figures_dir"]) / "val_errors.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)

    pairs = wrong.groupby(["label", "predicted"]).size().sort_values(ascending=False)
    write_json(
        project_path(cfg["paths"]["reports_dir"]) / "val_errors.json",
        {
            "errors": len(wrong),
            "images": len(val),
            "accuracy": float((val["predicted"] == val["label"]).mean()),
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
    print(f"{len(wrong)} validation errors -> {out}")


if __name__ == "__main__":
    main()
