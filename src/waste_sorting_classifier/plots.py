"""Report figures (matplotlib only, no PyTorch)."""

from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion(
    cm: np.ndarray, classes: list[str], path: Path, title: str = "Validation confusion matrix"
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_reliability(table: dict, path: Path, title: str) -> None:
    """Reliability diagram from metrics.reliability(): accuracy against confidence."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="perfect calibration")
    conf = [b["confidence"] for b in table["bins"]]
    acc = [b["accuracy"] for b in table["bins"]]
    sizes = [20 + 3 * b["photos"] for b in table["bins"]]
    ax.scatter(conf, acc, s=sizes, color="#2f7d5b", alpha=0.8, label="bins (size = photos)")
    ax.plot(conf, acc, color="#2f7d5b", linewidth=1)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="mean confidence", ylabel="accuracy",
           title=f"{title} (ECE {table['ece']:.3f})")  # fmt: skip
    ax.legend(loc="upper left", fontsize=8, markerscale=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
