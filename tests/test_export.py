"""ONNX export of an untrained checkpoint saved the way train.py saves it."""

import numpy as np
import pytest
import torch

from waste_sorting_classifier.export import compare, export_onnx, load_checkpoint
from waste_sorting_classifier.model import build_model


@pytest.mark.parametrize("arch", ["resnet18", "efficientnet_b0"])
def test_export_matches_pytorch(tmp_path, arch) -> None:
    torch.manual_seed(0)
    classes = ["a", "b", "c"]
    model = build_model(arch, len(classes), pretrained=False).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": arch}, checkpoint)

    loaded, loaded_classes, loaded_arch = load_checkpoint(checkpoint)
    assert (loaded_classes, loaded_arch) == (classes, arch)
    onnx_path = tmp_path / "model.onnx"
    export_onnx(loaded, onnx_path)

    batch = np.random.default_rng(0).normal(size=(3, 3, 224, 224)).astype(np.float32)
    result = compare(loaded, onnx_path, batch)  # also proves a batch of 3 works (dynamic axis)
    assert result["same_predictions"]
    assert result["max_abs_logit_diff"] < 1e-4
