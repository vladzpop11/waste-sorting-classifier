"""Download TrashNet (resized, 512x384) and RealWaste (524x524), verify them, index every image.

Each archive must match the SHA-256 pinned in configs/config.yaml, and the number of
images per class must match the published counts. Byte-identical files filed under two
different classes (3 pairs in TrashNet) have no knowable true label and are dropped.
RealWaste's classes are mapped onto ours (config: realwaste.classes); the classes we do
not use are left out. Writes data/processed/images.csv: relative path, class, file
SHA-256, size, source and original class per image.
"""

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from waste_sorting_classifier.artifacts import sha256, verify_sha256
from waste_sorting_classifier.config import load_config, project_path

ARCHIVE = "dataset-resized.zip"


def download(url: str, raw_dir: Path, expected_sha256: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE
    if not archive.exists():
        tmp = archive.with_name(ARCHIVE + ".download")
        with urllib.request.urlopen(url, timeout=120) as resp:
            tmp.write_bytes(resp.read())
        tmp.replace(archive)
    verify_sha256(archive, expected_sha256)
    return archive


def extract(archive: Path, raw_dir: Path) -> Path:
    """Extract image files only, skipping macOS metadata, into raw_dir/images/<class>/."""
    out = raw_dir / "images"
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or "__MACOSX" in parts or not info.filename.lower().endswith(".jpg"):
                continue
            target = out / parts[-2] / parts[-1]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
    return out


def extract_realwaste(cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """RealWaste: archive checked, published counts checked, used classes copied to
    images/realwaste/<our class>/ (file names without spaces). -> its index rows."""
    rw = cfg["realwaste"]
    archive = raw_dir / "realwaste.zip"
    if not (archive.exists() and sha256(archive) == rw["sha256"]):
        from waste_sorting_classifier.fetch_model import fetch

        fetch(rw["url"], archive, rw["sha256"], timeout=1800)
    out = raw_dir / "images" / rw["folder"]
    rows, counts = [], {}
    with zipfile.ZipFile(archive) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            parts = Path(info.filename).parts
            if info.is_dir() or not info.filename.lower().endswith(".jpg"):
                continue
            original = parts[-2]
            counts[original] = counts.get(original, 0) + 1
            label = rw["classes"].get(original)
            if label is None:
                continue
            target = out / label / parts[-1].replace(" ", "_")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
            with Image.open(target) as image:
                rows.append({"path": target.relative_to(raw_dir / "images").as_posix(),
                             "label": label, "sha256": sha256(target), "width": image.width,
                             "height": image.height, "source": "realwaste",
                             "original_label": original})  # fmt: skip
    if counts != rw["expected_counts"]:
        raise ValueError(
            f"RealWaste counts {counts} differ from the published {rw['expected_counts']}"
        )
    return pd.DataFrame(rows)


def build_index(images_dir: Path, classes: list[str]) -> pd.DataFrame:
    rows = []
    for label in classes:
        for path in sorted((images_dir / label).glob("*.jpg")):
            with Image.open(path) as image:
                rows.append(
                    {
                        "path": path.relative_to(images_dir).as_posix(),
                        "label": label,
                        "sha256": sha256(path),
                        "width": image.width,
                        "height": image.height,
                    }
                )
    return pd.DataFrame(rows)


def check_counts(index: pd.DataFrame, expected: dict[str, int]) -> None:
    counts = index["label"].value_counts().to_dict()
    if counts != expected:
        raise ValueError(f"image counts {counts} differ from the published counts {expected}")


def drop_conflicting_duplicates(index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove files whose identical bytes appear under more than one class."""
    labels_per_file = index.groupby("sha256")["label"].transform("nunique")
    conflicting = labels_per_file > 1
    return index[~conflicting].reset_index(drop=True), index[conflicting]


def main() -> None:
    cfg = load_config()
    d = cfg["data"]
    raw_dir = project_path(d["raw_dir"])
    archive = download(d["url"], raw_dir, d["sha256"])
    images_dir = extract(archive, raw_dir)
    index = build_index(images_dir, d["classes"])
    check_counts(index, d["expected_counts"])
    index, dropped = drop_conflicting_duplicates(index)
    index = index.assign(source="trashnet", original_label=index["label"])
    realwaste = extract_realwaste(cfg, raw_dir)
    if set(realwaste["sha256"]) & set(index["sha256"]):
        raise ValueError("a RealWaste file is byte-identical to a TrashNet file")
    index = pd.concat([index, realwaste], ignore_index=True)

    out = project_path(d["processed_dir"]) / "images.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out, index=False)
    print(f"archives' SHA-256 verified; published counts match; {len(dropped)} images dropped")
    print(f"(identical files under two classes: {sorted(dropped['path'])})")
    print(f"{len(index)} images -> {out}")
    print(pd.crosstab(index["label"], index["source"], margins=True).to_string())


if __name__ == "__main__":
    main()
