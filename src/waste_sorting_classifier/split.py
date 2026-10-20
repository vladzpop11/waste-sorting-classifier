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
import sys
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


def split_embeddings(cfg: dict, split: pd.DataFrame) -> np.ndarray:
    """The embeddings `make split` saved, one row per row of `split` (matched by path:
    embeddings.npy follows data/processed/images.csv, which can hold left-out photos)."""
    processed = project_path(cfg["data"]["processed_dir"])
    index = pd.read_csv(processed / "images.csv")
    embeddings = np.load(processed / "embeddings.npy")
    if len(embeddings) != len(index):
        raise SystemExit("embeddings.npy does not match images.csv: run `make split` first")
    row = pd.Series(np.arange(len(index)), index=index["path"])
    missing = set(split["path"]) - set(row.index)
    if missing:
        raise SystemExit(f"{len(missing)} split photos have no embedding: run `make split` first")
    return embeddings[row.loc[split["path"]].to_numpy()]


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


def place_realwaste(
    index: pd.DataFrame, embeddings: np.ndarray, frozen: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, list[str]]:
    """Version 4: TrashNet rows keep their frozen version 2 group and split. RealWaste rows
    are grouped among themselves like TrashNet was. A RealWaste group with a photo at or
    above the threshold to a TrashNet photo follows that photo's split; one tied to TrashNet
    photos in different splits goes to train, and its photos tied to validation or test
    photos are left out (they would be near-duplicates across the split). The other groups
    are assigned with StratifiedGroupKFold, except groups larger than chains_to_train_above
    (look-alike chains), which go to train. -> (rows kept, with group and split columns and
    their images.csv index; the left-out paths)."""
    threshold = cfg["split"]["similarity_threshold"]
    labels = index["label"].to_numpy()
    tn = (index["source"] == "trashnet").to_numpy()
    out = index.copy()
    fixed = frozen.set_index("path").loc[index.loc[tn, "path"]]
    out.loc[tn, "group"] = fixed["group"].to_numpy()
    out.loc[tn, "split"] = fixed["split"].to_numpy()

    rw = np.flatnonzero(~tn)
    groups = group_similar(embeddings[rw], labels[rw], threshold)
    out.loc[rw, "group"] = [f"realwaste-{g}" for g in groups]

    forced: dict[str, set[str]] = {}  # group -> TrashNet splits it is tied to
    tied: dict[int, set[str]] = {}  # photo -> TrashNet splits it is tied to
    for label in np.unique(labels):
        r = np.flatnonzero(~tn & (labels == label))
        t = np.flatnonzero(tn & (labels == label))
        sims = embeddings[r] @ embeddings[t].T
        for i, row in zip(r, sims, strict=True):
            for j in t[row >= threshold]:
                forced.setdefault(out.at[i, "group"], set()).add(out.at[j, "split"])
                tied.setdefault(i, set()).add(out.at[j, "split"])
    left_out = [
        i for i, splits in tied.items()
        if len(forced[out.at[i, "group"]]) > 1 and splits != {"train"}
    ]  # fmt: skip
    forced = {g: ({"train"} if len(s) > 1 else s) for g, s in forced.items()}
    sizes = out.loc[rw, "group"].value_counts()
    for group in sizes[sizes > cfg["split"]["chains_to_train_above"]].index:
        forced.setdefault(group, {"train"})  # look-alike chains train the model
    dropped = out.loc[left_out, "path"].tolist()
    out = out.drop(index=left_out)

    free = (out["source"] == "realwaste") & ~out["group"].isin(forced)
    out.loc[free, "split"] = assign_splits(
        out.loc[free, "label"].to_numpy(), out.loc[free, "group"].to_numpy(),
        cfg["split"]["n_folds"], cfg["seed"],
    )  # fmt: skip
    for group, splits in forced.items():
        out.loc[out["group"] == group, "split"] = next(iter(splits))
    return out, dropped


def main() -> None:
    cfg = load_config()
    index = pd.read_csv(project_path(cfg["data"]["processed_dir"]) / "images.csv")
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    threshold = cfg["split"]["similarity_threshold"]

    saved = project_path(cfg["data"]["processed_dir"]) / "embeddings.npy"
    if "--reuse-embeddings" in sys.argv[1:] and saved.exists():  # during development only
        embeddings = np.load(saved)
    else:
        embeddings = embed(index["path"].tolist(), images_dir)
        np.save(saved, embeddings)
    if "--embeddings-only" in sys.argv[1:]:  # for checking the committed split (CI)
        print(f"{len(embeddings)} embeddings saved")
        return
    frozen = pd.read_csv(project_path(cfg["split"]["trashnet_v2"]))
    placed, left_out = place_realwaste(index, embeddings, frozen, cfg)
    kept = placed.index.to_numpy()  # their rows in images.csv and embeddings.npy
    split = placed["split"].to_numpy()
    worst = max_cross_split_similarity(embeddings[kept], placed["label"].to_numpy(), split)
    if max(worst.values()) >= threshold:
        raise SystemExit(f"a cross-split pair reaches cosine {worst}; refusing to save")

    out = placed[["path", "label", "sha256", "group", "split", "source"]]
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
        "split_sizes_by_source": {
            src: rows["split"].value_counts().to_dict() for src, rows in out.groupby("source")
        },
        "left_out": left_out,
        "max_same_class_similarity_to_train": worst,
        "split_sha256": sha256(path),
    }
    write_json(project_path(cfg["paths"]["reports_dir"]) / "split_stats.json", stats)
    print(json.dumps(stats, indent=2))
    print(pd.crosstab([out["source"], out["label"]], out["split"], margins=True).to_string())


if __name__ == "__main__":
    main()
