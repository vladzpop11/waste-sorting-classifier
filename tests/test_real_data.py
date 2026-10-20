"""Checks on the real data and the committed split. Skipped until `make data` has run."""

import pytest

from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.split import load_split

CFG = load_config()
IMAGES = project_path(CFG["data"]["raw_dir"]) / "images"

pytestmark = pytest.mark.skipif(not IMAGES.exists(), reason="data not downloaded (make data)")


def test_committed_split_matches_the_images_and_keeps_groups_together() -> None:
    split = load_split(CFG)  # raises if any image differs from its recorded SHA-256
    assert len(split) == 2521
    assert (split.groupby("group")["split"].nunique() == 1).all()
    assert split["split"].value_counts().to_dict() == {"train": 1800, "val": 360, "test": 361}


def test_web_demo_records_the_evaluated_model() -> None:
    import json

    root = project_path("")
    web = json.loads((root / "web" / "model.json").read_text())
    evaluated = json.loads((root / "reports" / "test_metrics.json").read_text())
    assert web["onnx_sha256"] == evaluated["onnx_sha256"]
    assert json.loads((root / "models" / "model.json").read_text()) == web
