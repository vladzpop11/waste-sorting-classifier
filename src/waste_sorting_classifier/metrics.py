"""Evaluation statistics shared by the validation and test reports (no PyTorch)."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

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


def grouped_bootstrap(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_classes: int, seed: int
) -> dict:
    """95% intervals from resampling whole groups (photos of one object) with replacement.

    Photos of one object succeed or fail together, so resampling single photos would
    pretend there is more independent evidence than there is and give too-narrow intervals.
    """
    rng = np.random.default_rng(seed)
    labels = list(range(n_classes))
    names, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(names))]
    samples = {"accuracy": [], "macro_f1": [], "f1": [], "recall": []}
    for _ in range(N_BOOTSTRAP):
        i = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        samples["accuracy"].append(accuracy_score(y[i], pred[i]))
        samples["macro_f1"].append(f1_score(y[i], pred[i], labels=labels, average="macro",
                                            zero_division=0))  # fmt: skip
        _, recall, f1, support = precision_recall_fscore_support(
            y[i], pred[i], labels=labels, zero_division=0
        )
        absent = support == 0  # a class can be missing from a resample: no score for it
        samples["f1"].append(np.where(absent, np.nan, f1))
        samples["recall"].append(np.where(absent, np.nan, recall))

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.nanpercentile(values, 2.5, axis=0)),
                float(np.nanpercentile(values, 97.5, axis=0))]  # fmt: skip

    f1s, recalls = np.array(samples["f1"]), np.array(samples["recall"])
    return {
        "method": f"{N_BOOTSTRAP} resamples of whole groups ({len(members)} groups)",
        "accuracy": interval(np.array(samples["accuracy"])),
        "macro_f1": interval(np.array(samples["macro_f1"])),
        "per_class_f1": [interval(f1s[:, k]) for k in labels],
        "per_class_recall": [interval(recalls[:, k]) for k in labels],
    }
