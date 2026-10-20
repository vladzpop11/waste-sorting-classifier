"""Evidence for the grouping threshold: example pairs per similarity band, and how close
validation/test photos get to training photos. Uses data/processed/embeddings.npy (written
by split.py). Writes reports/figures/similarity_pairs.png and reports/data_stats.json.
"""

import numpy as np
import pandas as pd
from PIL import Image

from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path

BANDS = [(0.95, 1.01), (0.90, 0.95), (0.88, 0.90), (0.86, 0.88), (0.84, 0.86), (0.80, 0.84)]
PER_BAND = 4


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    split = pd.read_csv(project_path(cfg["split"]["file"]))
    emb = np.load(project_path(cfg["data"]["processed_dir"]) / "embeddings.npy")
    labels = split["label"].to_numpy()
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    sims = emb @ emb.T
    same = labels[:, None] == labels[None, :]
    iu = np.triu_indices(len(labels), 1)
    pair_sim, pair_same = sims[iu], same[iu]

    rng = np.random.default_rng(cfg["seed"])
    fig, axes = plt.subplots(len(BANDS), PER_BAND, figsize=(PER_BAND * 3.2, len(BANDS) * 1.5))
    band_counts = {}
    for row, (lo, hi) in enumerate(BANDS):
        in_band = np.flatnonzero((pair_sim >= lo) & (pair_sim < hi) & pair_same)
        band_counts[f"{lo:.2f}-{min(hi, 1):.2f}"] = int(len(in_band))
        picks = rng.choice(in_band, size=min(PER_BAND, len(in_band)), replace=False)
        for col in range(PER_BAND):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(picks):
                continue
            a, b = iu[0][picks[col]], iu[1][picks[col]]
            pair = [Image.open(images_dir / split.path[i]).resize((160, 120)) for i in (a, b)]
            canvas = Image.new("RGB", (324, 120), "white")
            canvas.paste(pair[0], (0, 0))
            canvas.paste(pair[1], (164, 0))
            ax.imshow(canvas)
            ax.set_title(f"{pair_sim[picks[col]]:.3f}  {labels[a]}", fontsize=8)
        band = f"{lo:.2f}–{min(hi, 1):.2f}"
        axes[row, 0].text(
            -0.08, 0.5, band, transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=9
        )
    fig.suptitle("Same-class photo pairs by cosine similarity of ImageNet ResNet18 features")
    fig.tight_layout()
    fig.savefig(project_path(cfg["paths"]["figures_dir"]) / "similarity_pairs.png", dpi=100)
    plt.close(fig)

    train = (split["split"] == "train").to_numpy()
    nearest = {}
    for name in ["val", "test"]:
        rows = np.flatnonzero(split["split"] == name)
        best = np.where(same[rows][:, train], sims[rows][:, train], -1).max(1)
        nearest[name] = {f">={t}": int((best >= t).sum()) for t in [0.80, 0.83, 0.84, 0.85, 0.86]}
        nearest[name]["photos"] = len(rows)
    stats = {"same_class_pairs_per_band": band_counts, "nearest_train_photo": nearest}
    write_json(project_path(cfg["paths"]["reports_dir"]) / "data_stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
