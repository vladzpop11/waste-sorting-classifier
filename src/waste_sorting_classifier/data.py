"""Download TrashNet (resized, 512x384), verify it, and index every image.

The archive must match the SHA-256 pinned in configs/config.yaml, and the number of
images per class must match the published counts. Byte-identical files filed under two
different classes (3 pairs in TrashNet) have no knowable true label and are dropped.
Writes data/processed/images.csv: relative path, class and file SHA-256 per image.
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

    out = project_path(d["processed_dir"]) / "images.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out, index=False)
    print(f"archive SHA-256 verified; published counts match; {len(dropped)} images dropped")
    print(f"(identical files under two classes: {sorted(dropped['path'])})")
    print(f"{len(index)} images -> {out}")
    print(index["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
