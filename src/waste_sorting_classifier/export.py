"""Export a trained checkpoint to ONNX and prove the export gives the same answers.

Loads model.pt with weights_only=True (no arbitrary pickles), exports with a dynamic batch
axis, checks the graph with onnx.checker, then runs the same validation photos through
PyTorch and ONNX Runtime: every predicted class must match and the logits must agree within
a tolerance. Writes models/model.onnx and models/model.json (classes, preprocessing,
hashes, the equivalence result), which is all the API and the demo need.

The ONNX file has two outputs: `logits`, and `features`, the 1280 numbers the final layer
reads (the familiarity check compares them with the training photos'). `features` is an
extra output of an existing tensor: the prediction path is unchanged. When the evaluated
file (reports/test_metrics.json) is in models/ before exporting, every training and
validation photo is run through both files to show the logits are the same.
"""

import argparse
import inspect
import json
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
    add_features_output(path)


def add_features_output(path: Path) -> None:
    """Expose the input of the final Gemm (the pooled features) as a second output."""
    graph = onnx.load(str(path))
    final = next(n for n in graph.graph.node if "logits" in n.output)
    graph.graph.node.append(onnx.helper.make_node("Identity", [final.input[0]], ["features"]))
    width = graph.graph.initializer
    dims = next(i.dims[1] for i in width if i.name == final.input[1])  # Gemm weight (out, in)
    graph.graph.output.append(onnx.helper.make_tensor_value_info(
        "features", onnx.TensorProto.FLOAT, ["batch", dims]))  # fmt: skip
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def same_as_evaluated(new: Path, evaluated: Path, batches: list[np.ndarray]) -> dict:
    """Logits of the new file against the evaluated one, on every given photo."""
    sessions = [
        ort.InferenceSession(str(f), providers=["CPUExecutionProvider"]) for f in (new, evaluated)
    ]
    diff, same, n = 0.0, True, 0
    for batch in batches:
        a, b = (s.run(["logits"], {"image": batch})[0] for s in sessions)
        diff = max(diff, float(np.abs(a - b).max()))
        same &= bool((a.argmax(1) == b.argmax(1)).all())
        n += len(batch)
    return {"images": n, "max_abs_logit_diff": diff, "same_predictions": same}


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
    p.add_argument("checkpoint", type=Path, nargs="?", default=None,
                   help="model.pt from a training run (default: the one chosen in "
                        "reports/model_selection.json, SHA-256 checked)")  # fmt: skip
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = load_config()
    if args.checkpoint is None:
        selection = json.loads(
            (project_path(cfg["paths"]["reports_dir"]) / "model_selection.json").read_text()
        )
        args.checkpoint = project_path(selection["chosen_checkpoint"])
        if not args.checkpoint.exists():
            raise SystemExit(f"{args.checkpoint} is missing: download the training run first")
        if sha256(args.checkpoint) != selection["chosen_checkpoint_sha256"]:
            raise SystemExit(f"{args.checkpoint} is not the checkpoint model_selection.json chose")
    out = args.out_dir or project_path(cfg["paths"]["models_dir"])
    out.mkdir(parents=True, exist_ok=True)
    images_dir = args.images_dir or project_path(cfg["data"]["raw_dir"]) / "images"

    model, classes, arch = load_checkpoint(args.checkpoint)
    onnx_path = out / "model.onnx"
    test = (
        json.loads((project_path(cfg["paths"]["reports_dir"]) / "test_metrics.json").read_text())
        if (project_path(cfg["paths"]["reports_dir"]) / "test_metrics.json").exists()
        else {}
    )
    evaluated_sha = test.get("onnx_sha256")
    evaluated = out / "evaluated.onnx"
    if onnx_path.exists() and sha256(onnx_path) == evaluated_sha:
        onnx_path.replace(evaluated)  # kept until the comparison below
    export_onnx(model, onnx_path)

    val = load_split(cfg, images_dir=images_dir).query("split == 'val'").sort_values("path")
    batch = np.stack([pp.preprocess(pp.load_rgb(images_dir / p)) for p in val["path"][:N_CHECK]])
    check = compare(model, onnx_path, batch)
    if not check["same_predictions"] or check["max_abs_logit_diff"] > TOLERANCE:
        onnx_path.unlink()
        raise SystemExit(f"ONNX export does not match PyTorch: {check}")

    matches_evaluated = None
    if evaluated.exists():
        split = load_split(cfg, images_dir=images_dir).query("split != 'test'").sort_values("path")
        paths = list(split["path"])
        batches = (np.stack([pp.preprocess(pp.load_rgb(images_dir / p)) for p in paths[i : i + 64]])
                   for i in range(0, len(paths), 64))  # fmt: skip
        matches_evaluated = same_as_evaluated(onnx_path, evaluated, batches)
        evaluated.unlink()
        if (
            not matches_evaluated["same_predictions"]
            or matches_evaluated["max_abs_logit_diff"] > TOLERANCE
        ):
            raise SystemExit(
                f"the new file does not reproduce the evaluated model: {matches_evaluated}"
            )

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
        "features": {
            "name": "features",
            "size": int(
                onnx.load(str(onnx_path)).graph.output[1].type.tensor_type.shape.dim[1].dim_value
            ),
        },
        "opset": OPSET,
        "exported_with": {"torch": torch.__version__, "onnx": onnx.__version__},
        "checkpoint_sha256": sha256(args.checkpoint),
        "onnx_sha256": sha256(onnx_path),
        "equivalence_check": check,
        # The file evaluated on the test set, and the proof this one gives the same logits
        # (training and validation photos; the test photos are not used again).
        "evaluated_onnx_sha256": evaluated_sha,
        "matches_evaluated": matches_evaluated,
    }
    write_json(out / "model.json", meta)
    print(f"exported {arch} -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"equivalence on {check['images']} validation photos: same predictions "
        f"{check['same_predictions']}, max |logit diff| {check['max_abs_logit_diff']:.2e}"
    )
    if matches_evaluated:
        print(f"against the evaluated file on {matches_evaluated['images']} photos: "
              f"max |logit diff| {matches_evaluated['max_abs_logit_diff']:.2e}")  # fmt: skip
    return meta


if __name__ == "__main__":
    main()
