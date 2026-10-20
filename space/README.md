---
title: Waste Sorting Classifier
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
license: mit
short_description: Classify a photo of waste into six recycling categories
---

# Waste sorting classifier

Upload a photo of a single waste item: the model returns the probability of each of six recycling categories (cardboard, glass, metal, paper, plastic, trash).

- **Model:** EfficientNet-B0 fine-tuned on [TrashNet](https://github.com/garythung/trashnet), exported to ONNX and run with ONNX Runtime on CPU.
- **Held-out test result:** 95.6% accuracy (95% CI 93.4–97.5%), macro F1 0.945. The weakest class is trash (F1 0.87).
- **Scope:** TrashNet photos show one object on a plain white background. Busy scenes, several objects or unusual lighting are outside what the model was trained on.
- **Examples:** one validation photo per class, chosen as the first in each class that the model gets right. They show what the model handles, not its error rate.
- **Code, data handling and evaluation:** [github.com/vladzpop11/waste-sorting-classifier](https://github.com/vladzpop11/waste-sorting-classifier). The model file is checked against its recorded SHA-256 at startup.
