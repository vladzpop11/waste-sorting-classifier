"""Fixtures for checking the browser demo against the Python pipeline (`make check-web`).

python -m waste_sorting_classifier.web_fixtures OUT_DIR

- OUT_DIR/{i}.rgb, {i}.f32, index.json: every validation photo decoded to RGB, its Python
  preprocessing tensor and the model's logits (web/check_preprocess.mjs, web/check_page.cjs).
- OUT_DIR/edge/ and edge.json: unusual uploads made from one validation photo, each with what
  the Python pipeline does with it (logits, or refused): every EXIF orientation in JPEG, PNG
  and WebP, transparency, 16-bit, greyscale, CMYK, a colour profile, MPO, progressive JPEG,
  and inputs that must be refused (web/check_page.cjs).
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.split import load_split

ICC_PROFILE = Path("/System/Library/ColorSync/Profiles/Display P3.icc")  # macOS only


def _encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **kwargs)
    return buffer.getvalue()


def _exif(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif.tobytes()


def edge_cases(photo: Image.Image) -> dict[str, bytes]:
    """name -> file bytes. The extension in the name is only for humans: the page sniffs."""
    w, h = photo.size
    cases = {}
    for k in range(1, 9):
        cases[f"orient{k}.jpg"] = _encode(photo, "JPEG", quality=92, exif=_exif(k))
        cases[f"orient{k}.png"] = _encode(photo, "PNG", exif=_exif(k))
        cases[f"orient{k}.webp"] = _encode(photo, "WEBP", lossless=True, exif=_exif(k))
    alpha = Image.linear_gradient("L").resize((w, h)).rotate(90)  # 0 (left) .. 255 (right)
    rgba = photo.copy()
    rgba.putalpha(alpha)
    cases["rgba.png"] = _encode(rgba, "PNG")
    cases["rgba.webp"] = _encode(rgba, "WEBP", lossless=True)
    la = photo.convert("L")
    la.putalpha(alpha)
    cases["la.png"] = _encode(la, "PNG")
    palette = photo.quantize(64)
    cases["palette-transparent.png"] = _encode(palette, "PNG", transparency=0)
    grey16 = (np.asarray(photo.convert("L"), dtype=np.uint16) * 257).astype(np.uint16)
    cases["grey16.png"] = _encode(Image.fromarray(grey16, mode="I;16"), "PNG")
    cases["grey.jpg"] = _encode(photo.convert("L"), "JPEG", quality=92)
    cases["cmyk.jpg"] = _encode(photo.convert("CMYK"), "JPEG", quality=92)
    cases["progressive.jpg"] = _encode(photo, "JPEG", quality=92, progressive=True)
    cases["lossy.webp"] = _encode(photo, "WEBP", quality=90)
    cases["mpo.jpg"] = _encode(photo, "MPO", save_all=True, append_images=[photo.rotate(180)])
    if ICC_PROFILE.exists():
        cases["display-p3.jpg"] = _encode(photo, "JPEG", quality=92,
                                          icc_profile=ICC_PROFILE.read_bytes())  # fmt: skip
    # Must be refused.
    cases["too-elongated.png"] = _encode(Image.new("RGB", (2600, 200), "white"), "PNG")
    cases["too-large.png"] = _encode(Image.new("L", (6000, 4500)), "PNG")
    cases["gif.gif"] = _encode(photo, "GIF")
    cases["truncated.jpg"] = cases["orient1.jpg"][: len(cases["orient1.jpg"]) // 2]
    cases["text.jpg"] = b"this is not an image\n" * 10
    return cases


def python_result(model: OnnxClassifier, data: bytes) -> dict:
    """What the Python pipeline (the API's path) does with an upload."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in ("JPEG", "MPO", "PNG", "WEBP"):
                raise ValueError(f"format {probe.format}")
            pp.check_size(*probe.size)
        x = pp.preprocess(pp.load_rgb(io.BytesIO(data)))
    except (OSError, ValueError, SyntaxError) as err:
        return {"refused": str(err)}
    return {"logits": model.session.run(["logits"], {"image": x[np.newaxis]})[0][0].tolist()}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    model = OnnxClassifier(project_path(cfg["paths"]["models_dir"]))
    images_dir = project_path(cfg["data"]["raw_dir"]) / "images"
    val = load_split(cfg).query("split == 'val'").sort_values("path")
    index = []
    for i, path in enumerate(val["path"]):
        image = pp.load_rgb(images_dir / path)
        x = pp.preprocess(image)
        logits = model.session.run(["logits"], {"image": x[np.newaxis]})[0][0]
        (out / f"{i}.rgb").write_bytes(np.asarray(image, dtype=np.uint8).tobytes())
        (out / f"{i}.f32").write_bytes(x.astype("<f4").tobytes())
        index.append({"file": i, "path": path, "width": image.width, "height": image.height,
                      "logits": logits.tolist()})  # fmt: skip
    (out / "index.json").write_text(json.dumps({"classes": model.classes, "images": index}))
    print(f"{len(index)} validation photos -> {out}")

    edge_dir = out / "edge"
    edge_dir.mkdir(exist_ok=True)
    photo = pp.load_rgb(images_dir / val["path"].iloc[0])
    edges = []
    for name, data in edge_cases(photo).items():
        (edge_dir / name).write_bytes(data)
        result = python_result(model, data)
        if name.startswith("orient") and "logits" in result:
            # The same file with its orientation ignored: how far the check is from passing
            # if the page got orientation wrong.
            x = pp.preprocess(pp.to_rgb(Image.open(io.BytesIO(data))))
            unrotated = model.session.run(["logits"], {"image": x[np.newaxis]})[0][0]
            result["logits_ignoring_orientation"] = unrotated.tolist()
        edges.append({"name": name, **result})
    (out / "edge.json").write_text(json.dumps(edges))
    print(f"{len(edges)} edge cases -> {edge_dir}")


if __name__ == "__main__":
    main()
