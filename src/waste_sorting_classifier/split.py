"""Split the images once into train/validation/test, keeping photos of one object together.

TrashNet photographs many objects several times. A random split would put photos of the
same object in training and test, so the test score would partly measure recognising
familiar objects. Photos of one class are grouped when their ImageNet ResNet18 embeddings
are similar: every pair at cosine >= the threshold is joined (connected components), and
average-linkage clusters at the same threshold are merged in. Joining every such pair is
what guarantees that no pair at or above the threshold ends up on both sides of the split;
average linkage alone does not (version 1 of this split leaked; see reports/history/v1).
Groups are assigned to splits with StratifiedGroupKFold, and main() refuses to save a split
that has a cross-split pair at or above the threshold.

The split is computed once and committed to splits/split.csv: the embeddings can differ in
the last decimals between platforms, so it is not recomputed elsewhere. load_split()
checks every image against its recorded SHA-256.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold

from waste_sorting_classifier.artifacts import sha256, write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.preprocess import load_rgb, preprocess


def embed(paths: list[str], images_dir) -> np.ndarray:
    """L2-normalised ImageNet ResNet18 features (512-d) for each image."""
    import torch
    import torchvision

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = np.stack([preprocess(load_rgb(images_dir / p)) for p in paths[i : i + 64]])
            out.append(torch.nn.functional.normalize(model(torch.from_numpy(batch)), dim=1).numpy())
    return np.concatenate(out)


def group_similar(embeddings: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Group ids within each class: connected components of "cosine >= threshold" pairs,
    merged with average-linkage clusters at the same threshold."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    groups = np.empty(len(labels), dtype=object)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        n = len(idx)
        rows, cols = np.nonzero(embeddings[idx] @ embeddings[idx].T >= threshold)
        if n > 1:
            clusters = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average",
                distance_threshold=1 - threshold,
            ).fit_predict(embeddings[idx])  # fmt: skip
            first = {c: k for k, c in reversed(list(enumerate(clusters)))}
            rows = np.r_[rows, np.arange(n)]
            cols = np.r_[cols, [first[c] for c in clusters]]
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, component = connected_components(graph, directed=False)
        groups[idx] = [f"{label}-{c}" for c in component]
    return groups


def max_cross_split_similarity(
    embeddings: np.ndarray, labels: np.ndarray, split: np.ndarray
) -> dict[str, float]:
    """For val and test: the highest same-class cosine similarity to any training photo."""
    train = split == "train"
    out = {}
    for name in ["val", "test"]:
        rows = np.flatnonzero(split == name)
        sims = embeddings[rows] @ embeddings[train].T
        same = labels[rows][:, None] == labels[train][None, :]
        out[name] = float(np.where(same, sims, -1).max())
    return out


def assign_splits(labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold 0 -> test, fold 1 -> validation, the rest -> train; groups never straddle."""
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split = np.full(len(labels), "train", dtype=object)
    for k, (_, idx) in enumerate(folds.split(np.zeros(len(labels)), labels, groups)):
        if k == 0:
            split[idx] = "test"
        elif k == 1:
            split[idx] = "val"
    return split


def load_split(
    cfg: dict | None = None, verify: bool = True, images_dir: Path | None = None
) -> pd.DataFrame:
    """The committed split, with every image checked against its recorded SHA-256."""
    cfg = cfg or load_config()
    split = pd.read_csv(project_path(cfg["split"]["file"]))
    if verify:
        images_dir = images_dir or project_path(cfg["data"]["raw_dir"]) / "images"
        bad = [p for p, h in zip(split["path"], split["sha256"], strict=True)
               if sha256(images_dir / p) != h]  # fmt: skip
        if bad:
            raise ValueError(f"{len(bad)} images differ from the committed split, e.g. {bad[:3]}")
    return split


def main() -> None:
    cfg = load_config()
    index = pd.read_csv(project_path(cfg["data"]["processed_dir"]) / "images.csv")
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    labels = index["label"].to_numpy()
    threshold = cfg["split"]["similarity_threshold"]

    embeddings = embed(index["path"].tolist(), images_dir)
    np.save(project_path(cfg["data"]["processed_dir"]) / "embeddings.npy", embeddings)
    groups = group_similar(embeddings, labels, threshold)
    split = assign_splits(labels, groups, cfg["split"]["n_folds"], cfg["seed"])
    worst = max_cross_split_similarity(embeddings, labels, split)
    if max(worst.values()) >= threshold:
        raise SystemExit(f"a cross-split pair reaches cosine {worst}; refusing to save")

    out = index[["path", "label", "sha256"]].assign(group=groups, split=split)
    path = project_path(cfg["split"]["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    sizes = out["group"].value_counts()
    stats = {
        "threshold": threshold,
        "images": len(out),
        "groups": int(len(sizes)),
        "photos_in_multi_photo_groups": int(sizes[sizes > 1].sum()),
        "largest_group": int(sizes.max()),
        "split_sizes": out["split"].value_counts().to_dict(),
        "max_same_class_similarity_to_train": worst,
        "split_sha256": sha256(path),
    }
    write_json(project_path(cfg["paths"]["reports_dir"]) / "split_stats.json", stats)
    print(json.dumps(stats, indent=2))
    print(pd.crosstab(out["label"], out["split"], margins=True).to_string())


if __name__ == "__main__":
    main()
