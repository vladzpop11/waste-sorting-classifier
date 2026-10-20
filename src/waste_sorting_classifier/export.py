"""Export a trained checkpoint to ONNX and prove the export gives the same answers.

Loads model.pt with weights_only=True (no arbitrary pickles), exports with a dynamic batch
axis, checks the graph with onnx.checker, then runs the same validation photos through
PyTorch and ONNX Runtime: every predicted class must match and the logits must agree within
a tolerance. Writes models/model.onnx and models/model.json (classes, preprocessing,
hashes, the equivalence result), which is all the API and the demo need.
"""

import argparse
import inspect
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.artifacts import sha256, write_json
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.model import build_model
from waste_sorting_classifier.split import load_split

OPSET = 17
TOLERANCE = 1e-4
N_CHECK = 20


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, list[str], str]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(saved["arch"], len(saved["classes"]), pretrained=False)
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved["classes"], saved["arch"]


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # PyTorch >= 2.5 defaults to a new exporter (needs onnxscript)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, pp.CROP, pp.CROP),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        **kwargs,
    )
    onnx.checker.check_model(onnx.load(str(path)))


def compare(model: torch.nn.Module, onnx_path: Path, batch: np.ndarray) -> dict:
    with torch.no_grad():
        expected = model(torch.from_numpy(batch)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"image": batch})[0]
    return {
        "images": len(batch),
        "max_abs_logit_diff": float(np.abs(expected - actual).max()),
        "same_predictions": bool((expected.argmax(1) == actual.argmax(1)).all()),
        "tolerance": TOLERANCE,
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path, help="model.pt from a training run")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = load_config()
    out = args.out_dir or project_path(cfg["paths"]["models_dir"])
    out.mkdir(parents=True, exist_ok=True)
    images_dir = args.images_dir or project_path(cfg["data"]["raw_dir"]) / "images"

    model, classes, arch = load_checkpoint(args.checkpoint)
    onnx_path = out / "model.onnx"
    export_onnx(model, onnx_path)

    val = load_split(cfg, images_dir=images_dir).query("split == 'val'").sort_values("path")
    batch = np.stack([pp.preprocess(pp.load_rgb(images_dir / p)) for p in val["path"][:N_CHECK]])
    check = compare(model, onnx_path, batch)
    if not check["same_predictions"] or check["max_abs_logit_diff"] > TOLERANCE:
        onnx_path.unlink()
        raise SystemExit(f"ONNX export does not match PyTorch: {check}")

    meta = {
        "arch": arch,
        "classes": classes,
        "input": {"name": "image", "shape": ["batch", 3, pp.CROP, pp.CROP], "dtype": "float32"},
        "preprocessing": {
            "resize_shorter_side": pp.RESIZE,
            "center_crop": pp.CROP,
            "mean": pp.MEAN.tolist(),
            "std": pp.STD.tolist(),
            "exif_transpose": True,
        },  # fmt: skip
        "output": {"name": "logits", "apply": "softmax"},
        "opset": OPSET,
        "exported_with": {"torch": torch.__version__, "onnx": onnx.__version__},
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_sha256": sha256(onnx_path),
        "equivalence_check": check,
    }
    write_json(out / "model.json", meta)
    print(f"exported {arch} -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"equivalence on {check['images']} validation photos: same predictions "
        f"{check['same_predictions']}, max |logit diff| {check['max_abs_logit_diff']:.2e}"
    )
    return meta


if __name__ == "__main__":
    main()
