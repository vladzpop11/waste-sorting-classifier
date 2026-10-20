"""Checks on the real data and the committed split. Skipped until `make data` has run."""

import json

import numpy as np
import pytest

from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.split import load_split, max_cross_split_similarity

CFG = load_config()
IMAGES = project_path(CFG["data"]["raw_dir"]) / "images"
STATS = json.loads(project_path("reports/split_stats.json").read_text())

pytestmark = pytest.mark.skipif(not IMAGES.exists(), reason="data not downloaded (make data)")


def test_committed_split_matches_the_images_and_its_recorded_stats() -> None:
    split = load_split(CFG)  # raises if any image differs from its recorded SHA-256
    assert len(split) == STATS["images"] == 2521
    assert (split.groupby("group")["split"].nunique() == 1).all()
    assert split["split"].value_counts().to_dict() == STATS["split_sizes"]
    assert split["group"].nunique() == STATS["groups"]


def test_no_similar_pair_straddles_the_split() -> None:
    embeddings = project_path(CFG["data"]["processed_dir"]) / "embeddings.npy"
    if not embeddings.exists():
        pytest.skip("embeddings are written by make split")
    split = load_split(CFG, verify=False)
    worst = max_cross_split_similarity(
        np.load(embeddings), split["label"].to_numpy(), split["split"].to_numpy()
    )
    assert max(worst.values()) < CFG["split"]["similarity_threshold"]


def test_web_demo_records_the_evaluated_model() -> None:
    root = project_path("")
    metrics = root / "reports" / "test_metrics.json"
    if not metrics.exists():
        pytest.skip("no evaluated model yet")
    code = (root / "web" / "model-info.js").read_text()
    web = json.loads(code[code.index("{") : code.rindex("}") + 1])
    model = json.loads((root / "models" / "model.json").read_text())
    assert web["sha256"] == model["onnx_sha256"] == json.loads(metrics.read_text())["onnx_sha256"]
    assert web["classes"] == model["classes"]
    assert web["arch"] == model["arch"]


def test_demo_examples_are_validation_photos() -> None:
    from waste_sorting_classifier.artifacts import sha256

    split = load_split(CFG, verify=False).set_index("sha256")
    for example in sorted(project_path("web/examples").glob("*.jpg")):
        row = split.loc[sha256(example)]
        assert (row["split"], row["label"]) == ("val", example.stem), example.name
