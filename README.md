# Waste Sorting Classifier

[![CI](https://github.com/vladzpop11/waste-sorting-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/vladzpop11/waste-sorting-classifier/actions/workflows/ci.yml)

Classify a photo of a waste item into six recycling categories: cardboard, glass, metal, paper, plastic or trash. The model is a pre-trained network fine-tuned on a free Kaggle GPU, exported to ONNX and served from Docker with no PyTorch.

## Result in brief

| On 361 held-out test photos | Value (95% bootstrap CI) |
| --- | --- |
| Accuracy | **95.6%** (93.4–97.5%) |
| Macro F1 | **0.945** (0.914–0.970) |
| Weakest class | trash, F1 0.87 (20 photos) |
| Latency, one upload, Docker on a 2-core 2018 MacBook Air | 64 ms mean, 74 ms p95 |

- **Fine-tuning earns its GPU:** on validation, it beats the linear-probe baseline (frozen ImageNet features + logistic regression) by 18 accuracy points, 95.6% against 77.8%.
- **The test score is trustworthy for its setting:** photos of the same object are kept together when splitting, so the test set measures new objects, not familiar ones. See [Data](#data).
- **The setting is narrow:** one object on a plain background. Busy real-world scenes were never tested.

**Try it:** [huggingface.co/spaces/VladZpp/waste-sorting-classifier](https://huggingface.co/spaces/VladZpp/waste-sorting-classifier). The model runs in your browser, so photos never leave your device.

## How to run

Requires Python 3.12 via [uv](https://docs.astral.sh/uv/), `make` and Docker. `make help` lists every target.

```bash
make setup data       # dependencies, then the dataset (SHA-256 and class counts checked)
make fetch-model      # the evaluated ONNX model from the GitHub release (SHA-256 checked)
make docker-build docker-run
curl -s -X POST localhost:8000/predict -F file=@tests/example.jpg
make docker-stop
```

- **`POST /predict`** takes a JPEG, PNG or WebP upload (field `file`, up to 5 MB). It returns the label and all six probabilities.
- **Interactive docs:** `GET /docs`.
- **Refused inputs:** other formats, oversized or decompression-bomb images, and non-multipart requests all get an error with the same shape, `{"detail": [{"type", "loc", "msg"}]}`.
- **Other targets:** `make check` (Ruff and pytest), `make baseline`, `make smoke-train` (one CPU epoch), `make export`, `make errors`, `make latency`, `make serve` (the API without Docker), `make demo-local` (the Gradio demo), `make check-web` (the browser pipeline against Python; needs Node.js and Chrome).

## Data

[TrashNet](https://github.com/garythung/trashnet) (Thung & Yang, 2016), MIT license: 2,527 photos of single objects on a white background (cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137). `make data` downloads it from [Hugging Face](https://huggingface.co/datasets/garythung/trashnet) and checks the archive's SHA-256 and the published counts.

Two problems were found before any modelling:

1. **The same photo under two labels.** Three photos appear twice, byte for byte, under different classes (glass/metal, glass/plastic, glass/plastic). Their true label is unknowable, so all six files are dropped: 2,521 photos remain.
2. **The same object photographed several times, from different angles.** A random split would put one object's photos in both training and test, and the test score would partly measure recognising familiar objects.
   - **Grouping:** photos are grouped by the similarity of their ImageNet ResNet18 features, using average-linkage clustering within each class at cosine similarity 0.86.
   - **Choosing the threshold:** it was set by looking at pairs. Down to about 0.88 they are clearly the same object; below 0.86, simple linking chains unrelated photos together.
   - **Result:** 727 photos (29%) share a group with others; the largest group has 18.
   - **Splitting:** whole groups are assigned with StratifiedGroupKFold, giving **1,800 train, 360 validation and 361 test** photos, with every class at about 14% in validation and test.
   - **Pinning:** the split is committed in [splits/split.csv](splits/split.csv), and every image is checked against its recorded SHA-256 whenever it is loaded.

## Approach

1. **Preprocessing written once.** [preprocess.py](src/waste_sorting_classifier/preprocess.py) reproduces torchvision's evaluation transform without PyTorch (tested to 1e-6), and applies EXIF orientation. Validation, export, the API and the local Gradio demo all use it, so training and serving cannot drift apart.
2. **Training on a Kaggle T4 GPU** ([kaggle/train.ipynb](kaggle/train.ipynb)): the notebook clones this repository and runs the unchanged training module.
   - **Recipe:** ImageNet-pretrained backbone with a new head; augmentation (random crops, flips, colour jitter) for training only; AdamW with a cosine schedule and label smoothing; early stopping on validation loss.
   - **Tracking:** every epoch is logged to MLflow, using a SQLite store inside the run directory.
3. **Two architectures, one rule.** ResNet18 and EfficientNet-B0 were both trained, and the rule was fixed before training: higher validation macro F1 wins, and within 0.01 the faster model wins.
4. **Export on the Mac.** The chosen checkpoint is loaded with `weights_only=True` and exported to ONNX (opset 17).
   - **Equivalence:** on 20 validation photos the predictions are identical, with a maximum logit difference of 5.8e-6.
   - **Cross-check:** the ONNX model reproduces the GPU model's validation accuracy exactly (95.56%).
5. **One test evaluation,** of the exported ONNX file the API serves. Its SHA-256 is recorded, and `make evaluate` refuses to run a second time.
6. **A browser demo with the same numbers.** The public demo is a static page ([web/](web/)) that runs the same ONNX file with ONNX Runtime Web, and checks its SHA-256 in the browser first.
   - **Why static:** Hugging Face now charges for Gradio Spaces, and static Spaces are free.
   - **Preprocessing in JavaScript:** [preprocess.js](web/preprocess.js) reimplements the Python preprocessing, including Pillow's fixed-point bilinear resize and Python's round-half-to-even centre crop.
   - **Proof:** `make check-web` compares the two on all 360 validation photos. The input tensors are bit-identical, and predictions match in Node. The real page in headless Chrome, which also does its own JPEG decoding, gives the same prediction for every photo, with probabilities within 2e-6.

## Results

Validation (360 photos) ([reports/model_selection.json](reports/model_selection.json), [reports/baselines.json](reports/baselines.json)):

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Majority class | 23.3% | 0.063 |
| Linear probe, frozen ImageNet ResNet18 | 77.8% | 0.714 |
| ResNet18, fine-tuned | 93.1% | 0.915 |
| **EfficientNet-B0, fine-tuned (chosen)** | **95.6%** | **0.946** |

EfficientNet-B0 wins by 0.031 macro F1, and is also the faster model. Each trained in under 4 minutes on the T4. Both were still improving slightly at their last epoch (best epochs 21 and 24 of 25). No settings were tuned.

Test (361 photos, evaluated once) ([reports/test_metrics.json](reports/test_metrics.json)):

| Class | Precision | Recall | F1 | Photos |
| --- | --- | --- | --- | --- |
| cardboard | 0.966 | 0.983 | 0.974 | 58 |
| glass | 0.985 | 0.930 | 0.957 | 71 |
| metal | 0.948 | 0.948 | 0.948 | 58 |
| paper | 0.988 | 0.965 | 0.976 | 85 |
| plastic | 0.907 | 0.986 | 0.944 | 69 |
| trash | 0.895 | 0.850 | 0.872 | 20 |

Test accuracy (95.6%) and macro F1 (0.945) match validation (95.6%, 0.946).

## What it gets wrong

![Validation errors: 16 of 360](reports/figures/val_errors.png)

The 16 validation errors ([reports/val_errors.json](reports/val_errors.json)) fall into three groups:

- **Printed cardboard and paper,** in both directions (5 errors): boxes with printed photos, flattened flaps.
- **Shiny surfaces read as metal** (6 errors): green glass bottles, a foil wrapper filed as trash, a bubble mailer and a plastic lid.
- **Trash,** the smallest and most mixed class, with 4 of 20 validation photos wrong.

Most errors are uncertain (10 of 16 below 63% confidence), but some are not: the plastic lid is called metal at 92%.

## Platform notes

- **Intel Macs** stop at PyTorch 2.2.2 (which needs NumPy 1.x) and ONNX Runtime 1.23.2. The project pins those there and supports Python 3.12 only. Everywhere else gets current versions, and Linux uses PyTorch's CPU-only build.
- **Kaggle runs Python 3.13,** which those pins exclude, so the notebook runs the code from source (`PYTHONPATH=src`) instead of installing it. This was rehearsed on Python 3.13 in Docker.
- **The image serves with ONNX Runtime only:** no PyTorch, pandas or uv. It is 544 MB, of which 74 MB is SymPy, a dependency ONNX Runtime declares but does not need for inference.
- **MLflow:** browse a run's records with `uvx --from mlflow==3.17.1 mlflow server --backend-store-uri sqlite:///<run>/mlflow.db`.

## Security

- **PyTorch 2.2.2, the Intel-Mac pin, has known advisories,** including code execution through `torch.load` of a malicious file. No newer PyTorch exists for that platform.
- **Why the risk is contained:**
  - It is used only for local development on the author's machine.
  - It only ever loads model files produced by this project's own training runs, with `weights_only=True`.
  - It is never part of the served image.
- **CI audits** the dependencies it actually installs (Linux, current PyTorch) with `pip-audit`.
- **The API checks the model at startup:** it refuses a model file whose SHA-256 differs from `models/model.json`.

## Limitations

- **The photos are studio-like:** one object, a white background, good light. Real bins, mixed items and clutter were never tested, and accuracy there would likely be much lower.
- **The data is small.** The test set is 361 photos, so accuracy is uncertain by about ±2 points. Trash has 20 test photos.
- **The grouped split reduces leakage but cannot rule it out.** Photos of one object taken from very different angles may fall below the 0.86 similarity threshold and end up in different splits, which would make the test score slightly optimistic.
- **Validation chose the model and the stopping epoch.** The test set was used once, at the end.
- **Confidence is not calibrated.** Errors can be confident (92% on the plastic lid), and no calibration was measured.
- **The demo's examples were chosen:** they are the first validation photo per class that the model gets right. They show what it handles, not its error rate.

## Next steps

- Answer "unsure" below a confidence level, and measure calibration (reliability curve, temperature scaling).
- Collect or source photos of real bins and mixed scenes, to evaluate outside the studio setting.
- Train slightly longer, since both models were still improving, keeping selection on validation.
- Quantise the ONNX model for faster CPU inference, and drop SymPy from the image.

## License

Code: [MIT](LICENSE). Data: TrashNet, MIT, downloaded by the pipeline and not included in this repository.
