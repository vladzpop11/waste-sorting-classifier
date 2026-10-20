"""Download the evaluated ONNX model from the GitHub release and verify it before use.

The file must match the SHA-256 recorded in models/model.json (committed); any other file
is deleted. No partial file is ever left in place.
"""

import json
import shutil
import sys
import urllib.request
from pathlib import Path

from waste_sorting_classifier.artifacts import sha256
from waste_sorting_classifier.config import load_config, project_path


def fetch(url: str, dest: Path, expected_sha256: str, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".download")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        actual = sha256(tmp)
        if actual != expected_sha256:
            raise ValueError(f"{url} has SHA-256 {actual}, expected {expected_sha256}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    cfg = load_config()
    models = project_path(cfg["paths"]["models_dir"])
    meta_path = models / "model.json"
    if not meta_path.exists():
        sys.exit(f"{meta_path} is missing; it records which model to fetch")
    expected = json.loads(meta_path.read_text())["onnx_sha256"]
    dest = models / "model.onnx"
    if dest.exists() and sha256(dest) == expected:
        print(f"{dest} is already the evaluated model")
        return
    fetch(cfg["release"]["model_url"], dest, expected)
    print(f"downloaded and verified {dest} (SHA-256 {expected[:16]}…)")


if __name__ == "__main__":
    main()
