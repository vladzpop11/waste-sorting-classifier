"""Evidence for the grouping threshold, per data source: example pairs per similarity band,
and how close validation/test photos get to training photos. Uses the embeddings split.py
saved. Writes reports/figures/similarity_pairs.png (TrashNet),
reports/figures/similarity_pairs_realwaste.png and reports/data_stats.json.
"""

import numpy as np
import pandas as pd
from PIL import Image

from waste_sorting_classifier.artifacts import write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.split import split_embeddings

BANDS = [(0.95, 1.01), (0.90, 0.95), (0.88, 0.90), (0.86, 0.88), (0.84, 0.86), (0.80, 0.84)]
PER_BAND = 4
FIGURES = {
    "trashnet": ("similarity_pairs.png", (160, 120)),
    "realwaste": ("similarity_pairs_realwaste.png", (140, 140)),
}


def pairs_figure(rows: pd.DataFrame, emb: np.ndarray, cfg: dict, source: str) -> dict:
    """Sample same-class pairs per band for one source; -> pair counts per band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = rows["label"].to_numpy()
    paths = rows["path"].to_numpy()
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    name, (w, h) = FIGURES[source]
    sims = emb @ emb.T
    iu = np.triu_indices(len(labels), 1)
    pair_sim, pair_same = sims[iu], (labels[:, None] == labels[None, :])[iu]

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
            pair = [Image.open(images_dir / paths[i]).resize((w, h)) for i in (a, b)]
            canvas = Image.new("RGB", (2 * w + 4, h), "white")
            canvas.paste(pair[0], (0, 0))
            canvas.paste(pair[1], (w + 4, 0))
            ax.imshow(canvas)
            ax.set_title(f"{pair_sim[picks[col]]:.3f}  {labels[a]}", fontsize=8)
        band = f"{lo:.2f}–{min(hi, 1):.2f}"
        axes[row, 0].text(
            -0.08, 0.5, band, transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=9
        )
    title = {"trashnet": "TrashNet", "realwaste": "RealWaste"}[source]
    fig.suptitle(
        f"{title}: same-class photo pairs by cosine similarity of ImageNet ResNet18 features"
    )
    fig.tight_layout()
    fig.savefig(project_path(cfg["paths"]["figures_dir"]) / name, dpi=100)
    plt.close(fig)
    return band_counts


def main() -> None:
    cfg = load_config()
    split = pd.read_csv(project_path(cfg["split"]["file"]))
    emb = split_embeddings(cfg, split)
    labels = split["label"].to_numpy()
    train = (split["split"] == "train").to_numpy()
    stats = {}
    for source in ["trashnet", "realwaste"]:
        m = (split["source"] == source).to_numpy()
        bands = pairs_figure(split[m].reset_index(drop=True), emb[m], cfg, source)
        nearest = {}
        for name in ["val", "test"]:
            rows = np.flatnonzero(m & (split["split"] == name).to_numpy())
            sims = emb[rows] @ emb[train].T  # against every training photo, both sources
            same = labels[rows][:, None] == labels[train][None, :]
            best = np.where(same, sims, -1).max(1)
            nearest[name] = {
                f">={t}": int((best >= t).sum()) for t in [0.80, 0.83, 0.84, 0.85, 0.86]
            }
            nearest[name]["photos"] = len(rows)
        stats[source] = {"same_class_pairs_per_band": bands, "nearest_train_photo": nearest}
    write_json(project_path(cfg["paths"]["reports_dir"]) / "data_stats.json", stats)
    print(stats)


if __name__ == "__main__":
    main()
