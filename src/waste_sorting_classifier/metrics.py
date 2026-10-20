"""Evaluation statistics shared by the validation and test reports (no PyTorch)."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

N_BOOTSTRAP = 2000


def reliability(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Expected calibration error (ECE) and the reliability table, over equal-width
    confidence bins: does "80% confident" mean right 80% of the time?"""
    confidence = proba.max(1)
    correct = proba.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        inside = (confidence > lo) & (confidence <= hi)
        if not inside.any():
            continue
        acc, conf = float(correct[inside].mean()), float(confidence[inside].mean())
        ece += inside.mean() * abs(acc - conf)
        bins.append({"from": round(float(lo), 2), "to": round(float(hi), 2),
                     "photos": int(inside.sum()), "accuracy": acc, "confidence": conf})  # fmt: skip
    return {"ece": float(ece), "bins": bins}


def low_confidence_threshold(
    confidence: np.ndarray, correct: np.ndarray, target: float = 0.99
) -> dict:
    """The lowest confidence threshold (0.50, 0.55, ... 0.95) above which predictions were
    right at least `target` of the time; predictions below it get a low-confidence note."""
    grid = np.round(np.arange(0.50, 0.951, 0.05), 2)
    chosen = next((t for t in grid if correct[confidence >= t].mean() >= target), grid[-1])
    below = confidence < chosen
    return {
        "threshold": float(chosen),
        "target_accuracy_above": target,
        "accuracy_above": float(correct[~below].mean()) if (~below).any() else None,
        "accuracy_below": float(correct[below].mean()) if below.any() else None,
        "share_below": float(below.mean()),
    }


def bootstrap_intervals(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_classes: int, seed: int
) -> dict:
    """95% intervals from resampling with replacement, two ways:

    - per photo: treats every photo as independent. Too narrow when several photos show
      the same object, because they succeed or fail together;
    - per group: resamples whole similarity groups. Too wide when a group chains together
      different objects (the split's groups are connected components, and one can be large).
    The truth lies between. Macro F1 averages the classes present in each resample: a class
    a resample happens to leave out has no score, not a score of 0.
    """
    return {
        "per_photo": _bootstrap(y, pred, np.arange(len(y)), n_classes, seed),
        "per_group": _bootstrap(y, pred, groups, n_classes, seed),
    }


def _bootstrap(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_classes: int,
               seed: int) -> dict:  # fmt: skip
    rng = np.random.default_rng(seed)
    labels = list(range(n_classes))
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(names))]
    samples = {"accuracy": [], "macro_f1": [], "f1": [], "recall": []}
    for _ in range(N_BOOTSTRAP):
        i = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        _, recall, f1, support = precision_recall_fscore_support(
            y[i], pred[i], labels=labels, zero_division=0
        )
        absent = support == 0
        samples["accuracy"].append(accuracy_score(y[i], pred[i]))
        samples["macro_f1"].append(f1[~absent].mean())
        samples["f1"].append(np.where(absent, np.nan, f1))
        samples["recall"].append(np.where(absent, np.nan, recall))

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.nanpercentile(values, 2.5, axis=0)),
                float(np.nanpercentile(values, 97.5, axis=0))]  # fmt: skip

    f1s, recalls = np.array(samples["f1"]), np.array(samples["recall"])
    return {
        "resamples": N_BOOTSTRAP,
        "units": len(members),
        "accuracy": interval(np.array(samples["accuracy"])),
        "macro_f1": interval(np.array(samples["macro_f1"])),
        "per_class_f1": [interval(f1s[:, k]) for k in labels],
        "per_class_recall": [interval(recalls[:, k]) for k in labels],
    }
