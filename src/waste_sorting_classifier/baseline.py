"""Baselines that fine-tuning must beat, on the validation split.

- majority class: always predict the most common training class;
- linear probe: frozen ImageNet ResNet18 features + logistic regression. This measures how
  much fine-tuning the network adds over reusing what it already knows.
Writes reports/baselines.json. The test split is not used.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.split import embed, load_split


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    cfg = load_config()
    split = load_split(cfg)
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    train, val = split[split["split"] == "train"], split[split["split"] == "val"]

    majority = train["label"].mode()[0]
    results = {"majority class": scores(val["label"].to_numpy(), np.full(len(val), majority))}

    features = {name: embed(rows["path"].tolist(), images_dir) for name, rows in
                [("train", train), ("val", val)]}  # fmt: skip
    probe = LogisticRegression(C=1.0, max_iter=5000).fit(features["train"], train["label"])
    results["linear probe (frozen ImageNet ResNet18)"] = scores(
        val["label"].to_numpy(), probe.predict(features["val"])
    )
    write_json(project_path(cfg["paths"]["reports_dir"]) / "baselines.json", results)
    for name, s in results.items():
        print(f"{name:42s} accuracy {s['val_accuracy']:.4f}  macro F1 {s['val_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
