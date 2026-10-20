"""A tiny end-to-end training run on synthetic images (CPU, untrained network, 1 epoch)."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from waste_sorting_classifier import train
from waste_sorting_classifier.artifacts import sha256
from waste_sorting_classifier.config import DEFAULT_CONFIG


@pytest.fixture
def tiny_project(tmp_path, monkeypatch):
    """Two classes of synthetic photos, a split file and a config pointing at them."""
    rng = np.random.default_rng(0)
    images = tmp_path / "images"
    rows = []
    for label, colour in [("glass", (40, 160, 60)), ("paper", (200, 200, 220))]:
        (images / label).mkdir(parents=True)
        for i in range(8):
            pixels = np.clip(rng.normal(colour, 25, (60, 80, 3)), 0, 255).astype(np.uint8)
            path = images / label / f"{label}{i}.jpg"
            Image.fromarray(pixels).save(path)
            split = "val" if i < 2 else "train"
            rows.append(
                {
                    "path": f"{label}/{label}{i}.jpg",
                    "label": label,
                    "sha256": sha256(path),
                    "group": f"{label}-{i}",
                    "split": split,
                }
            )
    split_file = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(split_file, index=False)

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["split"]["file"] = str(split_file)  # absolute paths win over the project root
    cfg["data"]["classes"] = ["glass", "paper"]
    cfg["train"] |= {"batch_size": 4, "epochs": 1}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("WASTE_CONFIG", str(config))
    return tmp_path, images


def test_training_run_produces_every_output(tiny_project) -> None:
    tmp_path, images = tiny_project
    out = tmp_path / "run"
    metrics = train.main(
        [
            "--output-dir",
            str(out),
            "--images-dir",
            str(images),
            "--no-pretrained",
            "--device",
            "cpu",
        ]
    )
    for name in ["model.pt", "metrics.json", "history.json", "confusion_val.png", "mlflow.db"]:
        assert (out / name).exists(), name
    saved = torch.load(out / "model.pt", map_location="cpu")
    assert saved["classes"] == ["glass", "paper"] and saved["arch"] == "resnet18"
    assert json.loads((out / "metrics.json").read_text())["classes"] == ["glass", "paper"]
    assert 0 <= metrics["val_accuracy"] <= 1
    assert np.array(metrics["val_confusion"]).sum() == 4  # 2 validation photos per class
    with sqlite3.connect(out / "mlflow.db") as db:
        assert db.execute("select count(*) from runs").fetchone()[0] == 1


def test_training_refuses_images_that_differ_from_the_split(tiny_project) -> None:
    tmp_path, images = tiny_project
    Image.new("RGB", (80, 60)).save(images / "glass" / "glass0.jpg")  # tamper with one image
    with pytest.raises(ValueError, match="differ from the committed split"):
        train.main(["--output-dir", str(tmp_path / "run"), "--images-dir", str(images),
                    "--no-pretrained", "--device", "cpu"])  # fmt: skip
