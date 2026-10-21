"""Fixtures for checking the browser demo against the Python pipeline (`make check-web`).

python -m waste_sorting_classifier.web_fixtures OUT_DIR

- OUT_DIR/{i}.rgb, {i}.f32, index.json: every validation photo decoded to RGB, its Python
  preprocessing tensor and the model's logits (web/check_preprocess.mjs, web/check_page.cjs).
- OUT_DIR/edge/ and edge.json: unusual and hostile uploads (web_edge_cases.py), each with
  what the Python pipeline does with it: its decoded pixels (edge/NAME.rgb) and logits, or
  the reason it is refused (web/check_decode.mjs, web/check_page.cjs).
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.familiarity import Familiarity
from waste_sorting_classifier.familiarity_build import ood_photos
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.split import load_split
from waste_sorting_classifier.web_edge_cases import KNOWN_DIFFERENCES, edge_cases

ICC_PROFILE = Path("/System/Library/ColorSync/Profiles/Display P3.icc")  # macOS only


def python_result(
    model: OnnxClassifier, data: bytes, familiarity: Familiarity | None = None
) -> dict:
    """What the Python pipeline (the API's path) does with an upload."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in ("JPEG", "MPO", "PNG", "WEBP"):
                raise ValueError(f"format {probe.format}")
            pp.check_size(*probe.size)
        x = pp.preprocess(pp.load_rgb(io.BytesIO(data)))
    except Exception as err:  # the API refuses any file Pillow cannot read
        return {"refused": f"{type(err).__name__}: {err}"}
    logits, features = model.run(x[np.newaxis])
    result = {"logits": logits[0].tolist()}
    if familiarity is not None:
        result["score"] = float(familiarity.score(logits, features)[0])
    return result


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    model = OnnxClassifier(project_path(cfg["paths"]["models_dir"]))
    familiarity = Familiarity(project_path(cfg["paths"]["models_dir"]), model.meta)
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    val = load_split(cfg).query("split == 'val'").sort_values("path")
    index = []
    for i, path in enumerate(val["path"]):
        image = pp.load_rgb(images_dir / path)
        x = pp.preprocess(image)
        logits, features = model.run(x[np.newaxis])
        logits = logits[0]
        score = float(familiarity.score(logits[np.newaxis], features)[0])
        (out / f"{i}.rgb").write_bytes(np.asarray(image, dtype=np.uint8).tobytes())
        (out / f"{i}.f32").write_bytes(x.astype("<f4").tobytes())
        index.append({"file": i, "path": path, "width": image.width, "height": image.height,
                      "score": score,
                      "logits": logits.tolist()})  # fmt: skip
    record = {"classes": model.classes, "threshold": familiarity.threshold, "images": index}
    (out / "index.json").write_text(json.dumps(record))
    print(f"{len(index)} validation photos -> {out}")

    edge_dir = out / "edge"
    edge_dir.mkdir(exist_ok=True)
    photo = pp.load_rgb(images_dir / val["path"].iloc[0])
    icc = ICC_PROFILE.read_bytes() if ICC_PROFILE.exists() else None
    edges = []
    for name, data in edge_cases(photo, icc).items():
        (edge_dir / name).write_bytes(data)
        result = python_result(model, data, familiarity)
        if "logits" in result:
            # Python's decoded pixels, for an exact comparison (web/check_decode.mjs).
            image = pp.load_rgb(io.BytesIO(data))
            (edge_dir / f"{name}.rgb").write_bytes(np.asarray(image, dtype=np.uint8).tobytes())
            result.update(width=image.width, height=image.height)
        if name.startswith("orient") and "logits" in result:
            # The same file with its orientation ignored: how far the check is from passing
            # if the page got orientation wrong.
            x = pp.preprocess(pp.to_rgb(Image.open(io.BytesIO(data))))
            unrotated = model.session.run(["logits"], {"image": x[np.newaxis]})[0][0]
            result["logits_ignoring_orientation"] = unrotated.tolist()
        if name in KNOWN_DIFFERENCES:
            page, reason = KNOWN_DIFFERENCES[name]
            result["page_differs"] = {"page": page, "reason": reason}
        edges.append({"name": name, **result})
    # Everyday photos not used to choose the familiarity check (when `make familiarity-data`
    # has run): the page must flag exactly the ones the API flags.
    if (project_path(cfg["familiarity"]["ood_dir"]) / "imagenette2-160").exists():
        for path in ood_photos(cfg)[1][:30]:
            name = f"everyday-{path.parent.name}-{path.name}"
            data = path.read_bytes()
            (edge_dir / name).write_bytes(data)
            image = pp.load_rgb(io.BytesIO(data))
            (edge_dir / f"{name}.rgb").write_bytes(np.asarray(image, dtype=np.uint8).tobytes())
            edges.append({"name": name, "width": image.width, "height": image.height,
                          **python_result(model, data, familiarity)})  # fmt: skip
    (out / "edge.json").write_text(json.dumps(edges))
    refused = sum("refused" in e for e in edges)
    print(f"{len(edges)} edge cases ({refused} refused by Python) -> {edge_dir}")


if __name__ == "__main__":
    main()
