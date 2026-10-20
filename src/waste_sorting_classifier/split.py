"""Split the images once into train/validation/test, keeping photos of one object together.

TrashNet photographs many objects several times. A random split would put photos of the
same object in training and test, so the test score would partly measure recognising
familiar objects. Photos are grouped by average-linkage clustering (within each class) of
ImageNet ResNet18 embeddings, and groups are assigned to splits with StratifiedGroupKFold.

The split is computed once and committed to splits/split.csv: the embeddings can differ in
the last decimals between platforms, so it is not recomputed elsewhere. load_split()
checks every image against its recorded SHA-256.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedGroupKFold

from waste_sorting_classifier.artifacts import sha256
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
    """Group ids: average-linkage clusters of cosine similarity >= threshold, per class."""
    groups = np.empty(len(labels), dtype=object)
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() == 1:
            groups[mask] = [f"{label}-0"]
            continue
        clusters = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - threshold
        ).fit_predict(embeddings[mask])
        groups[mask] = [f"{label}-{c}" for c in clusters]
    return groups


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


def load_split(cfg: dict | None = None, verify: bool = True) -> pd.DataFrame:
    """The committed split, with every image checked against its recorded SHA-256."""
    cfg = cfg or load_config()
    split = pd.read_csv(project_path(cfg["split"]["file"]))
    if verify:
        images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
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

    embeddings = embed(index["path"].tolist(), images_dir)
    groups = group_similar(embeddings, labels, cfg["split"]["similarity_threshold"])
    split = assign_splits(labels, groups, cfg["split"]["n_folds"], cfg["seed"])

    out = index[["path", "label", "sha256"]].assign(group=groups, split=split)
    path = project_path(cfg["split"]["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    sizes = out["group"].value_counts()
    print(f"{len(sizes)} groups; {sizes[sizes > 1].sum()} photos in multi-photo groups; "
          f"largest {sizes.max()}")  # fmt: skip
    print(pd.crosstab(out["label"], out["split"], margins=True).to_string())
    print(f"-> {path} (SHA-256 {sha256(path)[:16]}…)")


if __name__ == "__main__":
    main()
