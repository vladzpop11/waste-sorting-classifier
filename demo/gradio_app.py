"""Local Gradio demo: classify a photo of a waste item into six recycling categories.

Run with `make demo-local` (after `make fetch-model`). It uses the same model files and
preprocessing as the API; the public demo is the static page in web/, because hosting a
Gradio Space on Hugging Face now needs a paid plan.
"""

from pathlib import Path

import gradio as gr
import numpy as np

from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.preprocess import load_rgb, preprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL = OnnxClassifier(ROOT / "models", threads=2)
EXAMPLES = sorted(str(p) for p in (ROOT / "web" / "examples").glob("*.jpg"))


def classify(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    proba = MODEL.predict_proba(preprocess(load_rgb(path))[np.newaxis])[0]
    return {c: float(p) for c, p in zip(MODEL.classes, proba, strict=True)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="filepath", label="Photo of one waste item"),
    outputs=gr.Label(num_top_classes=6, label="Recycling category"),
    examples=EXAMPLES,
    title="Waste sorting classifier",
    description=(
        "EfficientNet-B0 fine-tuned on TrashNet: cardboard, glass, metal, paper, plastic or "
        "trash. 94.3% accuracy on 366 test photos of single objects on a plain background, "
        "with no near-duplicate in training; busy real-world scenes are outside what it was "
        "trained on. "
        "[Code and evaluation](https://github.com/vladzpop11/waste-sorting-classifier)"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
