"""Write validation photos as decoded RGB bytes plus the Python reference results, for
web/check_preprocess.mjs to compare the browser preprocessing and model against.

python -m waste_sorting_classifier.web_fixtures OUT_DIR
"""

import json
import sys
from pathlib import Path

import numpy as np

from waste_sorting_classifier import preprocess as pp
from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.split import load_split


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


if __name__ == "__main__":
    main()
