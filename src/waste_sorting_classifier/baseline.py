"""Baselines that fine-tuning must beat, on the validation split.

- majority class: always predict the most common training class;
- linear probe: frozen ImageNet ResNet18 features + logistic regression. This measures how
  much fine-tuning the network adds over reusing what it already knows (features: the
  embeddings `make split` saved). Its regularisation
  strength C is tuned by 5-fold grouped cross-validation on the training photos only (groups
  never straddle folds), so the baseline is as strong as it fairly can be.
Writes reports/baselines.json. The test split is not used.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.split import load_split, split_embeddings

C_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def by_source(val, pred: np.ndarray) -> dict:
    """Scores on all validation photos, and per data source."""
    out = scores(val["label"].to_numpy(), pred)
    for source in sorted(val["source"].unique()):
        m = (val["source"] == source).to_numpy()
        out[source] = scores(val["label"].to_numpy()[m], pred[m])
    return out


def main() -> None:
    cfg = load_config()
    split = load_split(cfg)
    train, val = split[split["split"] == "train"], split[split["split"] == "val"]

    majority = train["label"].mode()[0]
    results = {"majority class": by_source(val, np.full(len(val), majority))}

    # The same frozen ResNet18 features the split was built from (make split saved them).
    features = {"train": split_embeddings(cfg, train), "val": split_embeddings(cfg, val)}
    search = GridSearchCV(
        LogisticRegression(max_iter=5000),
        {"C": C_GRID},
        scoring="f1_macro",
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg["seed"]),
    ).fit(features["train"], train["label"], groups=train["group"])
    results["linear probe (frozen ImageNet ResNet18)"] = {
        **by_source(val, search.predict(features["val"])),
        "C": search.best_params_["C"],
        "cv_macro_f1_by_C": dict(
            zip(
                map(str, C_GRID),
                search.cv_results_["mean_test_score"].round(4).tolist(),
                strict=True,
            )
        ),
    }
    write_json(project_path(cfg["paths"]["reports_dir"]) / "baselines.json", results)
    for name, s in results.items():
        extra = f"  (C={s['C']})" if "C" in s else ""
        for label, r in [(name, s), ("  trashnet", s["trashnet"]), ("  realwaste", s["realwaste"])]:
            acc, f1 = r["val_accuracy"], r["val_macro_f1"]
            print(f"{label:42s} accuracy {acc:.4f}  macro F1 {f1:.4f}{extra if r is s else ''}")


if __name__ == "__main__":
    main()
