"""Gradio demo: classify a photo of a waste item into six recycling categories.

The model (model.onnx) ships with this Space and is checked against the SHA-256 in
model.json at startup. The preprocessing and inference code come from the project's
GitHub repository at a pinned commit (see requirements.txt), so the demo runs exactly what
was evaluated.
"""

from pathlib import Path

import gradio as gr
import numpy as np

from waste_sorting_classifier.onnx_model import OnnxClassifier
from waste_sorting_classifier.preprocess import load_rgb, preprocess

HERE = Path(__file__).parent
MODEL = OnnxClassifier(HERE, threads=2)
EXAMPLES = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))


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
        "trash. 95.6% accuracy on held-out photos of single objects on a plain background; "
        "busy real-world scenes are outside what it was trained on. "
        "[Code and evaluation](https://github.com/vladzpop11/waste-sorting-classifier)"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
