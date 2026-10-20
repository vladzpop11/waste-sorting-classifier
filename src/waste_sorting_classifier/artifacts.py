"""Hashing and JSON records shared by the pipeline."""

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path} has SHA-256 {actual}, but configs/config.yaml pins {expected}. "
            "The file is not the one the recorded results were produced from; restore it, "
            "or update the pin knowingly."
        )


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
