# Waste Sorting Classifier

[![CI](https://github.com/vladzpop11/waste-sorting-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/vladzpop11/waste-sorting-classifier/actions/workflows/ci.yml)

Classify a photo of a waste item into six recycling categories: cardboard, glass, metal, paper, plastic or trash. The model is a pre-trained network fine-tuned on a free Kaggle GPU, exported to ONNX, and served two ways: from Docker with no PyTorch, and in the browser.

## Result in brief

| On 366 test photos with no near-duplicate in training | Value | 95% CI, per photo | 95% CI, per group |
| --- | --- | --- | --- |
| Accuracy | **94.3%** | 91.8–96.4% | 90.3–96.9% |
| Macro F1 | **0.936** | 0.905–0.963 | 0.905–0.965 |
| Weakest class: trash (19 photos), F1 | 0.889 | 0.762–0.974 | |
| Latency, one upload, Docker on a 2-core 2018 MacBook Air | 49 ms mean, 63 ms p95 | | |

- **The test set measures new objects.** Photos of the same object are kept on one side of the split, with a checked guarantee. A first version of the split leaked; it was rebuilt and the models retrained. See [Data](#data).
- **Fine-tuning earns its GPU.** On validation it beats a tuned linear probe on frozen ImageNet features by 11 accuracy points, 94.0% against 83.1%.
- **The setting is narrow:** one object on a plain background. On the 25 test photos least like any training photo of their class, accuracy falls to 68%.

**Try it:** [huggingface.co/spaces/VladZpp/waste-sorting-classifier](https://huggingface.co/spaces/VladZpp/waste-sorting-classifier). The model runs in your browser, so photos never leave your device.

## How to run

Requires Python 3.12 via [uv](https://docs.astral.sh/uv/), `make` and Docker. `make help` lists every target.

```bash
make setup data       # dependencies, then the dataset (SHA-256 and class counts checked)
make fetch-model      # the evaluated ONNX model from the GitHub release (SHA-256 checked)
make docker-build docker-run
curl -s --retry 10 --retry-connrefused --retry-delay 1 localhost:8000/health
curl -s -X POST localhost:8000/predict -F file=@tests/example.jpg
make docker-stop
```

- **`POST /predict`** takes a JPEG, PNG or WebP upload (field `file`, up to 5 MB, 25 megapixels, 10:1 aspect ratio). It returns the label and all six probabilities. Interactive docs: `GET /docs`.
- **Refused inputs** get an error with one shape, `{"detail": [{"type", "loc", "msg"}]}`: 413 too large, 415 not a multipart upload, 422 not a readable image or outside the limits, 503 busy (two predictions at a time).
- **Pipeline targets,** in order: `make split`, `make baseline`, `make smoke-train` (one CPU epoch), training on Kaggle, `make select`, `make export`, `make errors`, `make evaluate` (once), `make latency`, `make web`.
- **Checks:** `make check` (Ruff and pytest), `make docker-smoke`, `make check-web` (the browser against Python; needs Node.js and Chrome). CI runs all three.

## Data

[TrashNet](https://github.com/garythung/trashnet) (Thung & Yang, 2016), MIT license: 2,527 photos of single objects on a white background. `make data` downloads it from [Hugging Face](https://huggingface.co/datasets/garythung/trashnet) and checks the archive's SHA-256 and the published class counts. The full account is in [reports/data.md](reports/data.md).

1. **The same photo under two labels.** Three photos appear twice, byte for byte, under different classes. Their true label is unknowable, so all six files are dropped: 2,521 photos remain.
2. **The same object, photographed several times.** A random split would put one object's photos on both sides, and the test would partly measure recognising familiar objects.
   - **Finding them:** photos are compared by cosine similarity of ImageNet ResNet18 features, within each class. Looking at pairs in each similarity band put the threshold at 0.86 ([figure](reports/figures/similarity_pairs.png)).
   - **Grouping:** every same-class pair at ≥ 0.86 is joined (connected components), so no group can straddle the split.
   - **Splitting:** whole groups go to one side with StratifiedGroupKFold: **1,795 train, 360 validation and 366 test** photos.
   - **The guarantee:** `make split` refuses to save a split in which any validation or test photo has a same-class training photo at ≥ 0.86. The highest is 0.8599. The split is committed in [splits/split.csv](splits/split.csv), and every image is checked against its SHA-256 whenever it is loaded.
3. **Version 1 leaked.** The first split grouped by average-linkage clusters only. A review found 44 test photos with a training twin at ≥ 0.86. That model scored 100% on them and 95.0% on the rest, against 95.6% overall. Everything from version 1 is kept in [reports/history/v1](reports/history/v1).

## Approach

1. **Preprocessing written once.** [preprocess.py](src/waste_sorting_classifier/preprocess.py) reproduces torchvision's evaluation transform without PyTorch (tested to 1e-6). It also turns photos upright from EXIF, scales 16-bit images, flattens transparency onto white, and refuses oversized or very elongated images. Validation, export, the API and both demos use it.
2. **Training on a Kaggle T4 GPU** ([kaggle/train.ipynb](kaggle/train.ipynb)): the notebook clones a fixed tag of this repository and runs the unchanged training module.
   - **Recipe:** ImageNet-pretrained backbone with a new head; augmentation (random crops, flips, colour jitter) for training only; AdamW with a cosine schedule and label smoothing; early stopping on validation loss. No settings were tuned.
   - **Tracking:** every epoch is logged to MLflow, in a SQLite store inside each run's directory.
3. **Model choice by a rule fixed before training** ([configs/config.yaml](configs/config.yaml), `selection`). ResNet18 and EfficientNet-B0 are each trained with three seeds; the higher mean validation macro F1 wins, and within 0.01 the faster model wins. The deployed checkpoint is the winner's run with the lowest validation loss. `make select` applies the rule and records the decision with the checkpoint's SHA-256.
4. **Export on the Mac.** The selected checkpoint (its SHA-256 checked) is loaded with `weights_only=True` and exported to ONNX (opset 17).
   - **Equivalence:** on 20 validation photos the predictions are identical, with a maximum logit difference of 1.4e-6.
   - **Cross-check:** the ONNX model reproduces the GPU run's validation scores exactly (95.00%, macro F1 0.9398).
5. **One test evaluation,** of the exported ONNX file that the API serves. `make evaluate` checks that this file was exported from the selected checkpoint, and refuses to run a second time. It was first rehearsed on validation (`--rehearse`), so the single real run was not the first.
6. **A browser demo with the same numbers, in every browser.** The public demo is a static page ([web/](web/)) that runs the same ONNX file with ONNX Runtime Web.
   - **Decoding without the browser:** browsers decode images differently from Pillow. Safari uses its own JPEG decoder, and browsers apply colour profiles and premultiply transparency. So the page decodes images itself ([decode.js](web/decode.js)):
     - JPEG and WebP with WebAssembly builds of libjpeg-turbo (via MozJPEG) and libwebp, the libraries Pillow uses;
     - PNG with its own decoder ([png.js](web/png.js));
     - transparency flattened with Pillow's arithmetic;
     - EXIF and XMP orientation read exactly where Pillow reads them ([image-header.js](web/image-header.js)).
   - **Preprocessing in JavaScript:** [preprocess.js](web/preprocess.js) reproduces Pillow's fixed-point bilinear resize and Python's round-half-to-even centre crop.
   - **Proof:** `make check-web` (with `BROWSER=chrome`, `webkit` or `firefox`) runs three checks:
     - the page's decoding produces pixels identical to Python's on all 360 validation photos and on 109 unusual files: every EXIF orientation layout, 1- to 16-bit and interlaced PNG, transparency, MPO, progressive JPEG and more;
     - 10 damaged or hostile files are refused, as the API refuses them;
     - the real page, in Chrome, WebKit (Safari's engine) and Firefox, gives Python's prediction for all 360 photos, with probabilities within 3e-6.
   - **Documented difference:** CMYK JPEGs and animated WebP are refused by the page, which says so, although the API accepts them.
   - **What the checks caught:** relying on browser decoding would have failed this. Safari's own JPEG decoding changed 1 of 360 predictions, Firefox flattened transparency differently, and Chrome applied EXIF orientation even when told not to. A review also found a 67-byte PNG that froze the earlier page for five minutes; it is now refused in milliseconds and part of the test set.

## Results

Validation (360 photos) ([reports/model_selection.json](reports/model_selection.json), [reports/baselines.json](reports/baselines.json)):

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Majority class | 23.6% | 0.064 |
| Linear probe, frozen ImageNet ResNet18 (C tuned by grouped CV on training photos) | 83.1% | 0.809 |
| ResNet18, fine-tuned, mean of 3 seeds | 92.6% | 0.916 (sd 0.009) |
| **EfficientNet-B0, fine-tuned, mean of 3 seeds** | **94.0%** | **0.931** (sd 0.010) |
| EfficientNet-B0, deployed run (seed 44) | 95.0% | 0.940 |

- **The rule chose EfficientNet-B0** by 0.015 macro F1, more than the 0.01 margin, so speed did not decide it. The gap is about 1.5 seed standard deviations: suggestive, not conclusive (with three seeds each, a Welch t-test gives p = 0.13). The rule was fixed in advance, so the choice stands either way.
- **The deployed run's validation score is optimistic.** It is the best of three seeds on validation; the three-seed mean is the fairer estimate, and the test result agrees with it.
- **Training length:** five of the six runs had their best epoch at 22–24 of 25, so longer training might help a little.

Test (366 photos, evaluated once) ([reports/test_metrics.json](reports/test_metrics.json), [confusion matrix](reports/figures/confusion_test.png)):

| Class | Precision | Recall | F1 (95% CI per photo) | Photos |
| --- | --- | --- | --- | --- |
| cardboard | 1.000 | 0.982 | 0.991 (0.970–1.000) | 57 |
| glass | 0.955 | 0.887 | 0.920 (0.866–0.964) | 71 |
| metal | 0.930 | 0.914 | 0.922 (0.865–0.967) | 58 |
| paper | 0.976 | 0.952 | 0.964 (0.932–0.989) | 84 |
| plastic | 0.875 | 1.000 | 0.933 (0.891–0.968) | 77 |
| trash | 0.941 | 0.842 | 0.889 (0.762–0.974) | 19 |

**Two kinds of interval.** Resampling single photos treats each as independent evidence, which is too optimistic when photos come in families. Resampling whole groups is too cautious here: the 77 plastic test photos form one chained group of look-alikes ([reports/data.md](reports/data.md)). Both are reported.

**Accuracy depends on how familiar a photo looks.** Each test photo's similarity to its nearest same-class training photo (ImageNet features; always below 0.86 by construction):

| Nearest training photo | Test photos | Accuracy |
| --- | --- | --- |
| below 0.75 | 25 | 68.0% |
| 0.75–0.80 | 72 | 90.3% |
| 0.80–0.85 | 223 | 97.8% |
| 0.85–0.86 | 46 | 97.8% |

Most test photos have a fairly similar training photo, and those are classified well. Photos unlike any training photo of their class are much harder; that is the better guide to performance on new kinds of objects.

**Calibration** ([reliability diagram](reports/figures/reliability_test.png)): the model is underconfident, as label smoothing tends to make it (expected calibration error 0.085). Answers at 90% confidence or more were right 99.0% of the time on test (208 photos). The demo marks answers below 90% as less certain. That threshold was chosen on validation by a fixed rule: the lowest threshold above which validation accuracy reaches 99%.

## What it gets wrong

![Validation errors: 18 of 360](reports/figures/val_errors.png)

Most of the 18 validation errors ([reports/val_errors.json](reports/val_errors.json)) and 21 test errors fall into three groups:

- **Transparent and shiny items** confused among glass, plastic and metal: 10 validation and 11 test errors. Examples are green and clear bottles seen end-on, lids and a foil sweet bag.
- **Crumpled paper and wrappers called trash, and trash called plastic or metal:** 6 validation errors. Trash is the smallest and most mixed class.
- **Paper and plastic confused:** both directions on validation; on test, 3 paper photos called plastic.

The other 5 test errors are scattered (for example paper called metal, trash called paper).

Most errors are uncertain: 16 of 18 validation errors and 19 of 21 test errors were below 90% confidence. A few are confident: a green plastic bottle called glass at 92%.

## Platform notes

- **Intel Macs** stop at PyTorch 2.2.2 (which needs NumPy 1.x) and ONNX Runtime 1.23.2. The project pins those there and supports Python 3.12 only; pandas is held below 3 so that every platform runs the same major version. Everywhere else gets current versions, and Linux uses PyTorch's CPU-only build.
- **Kaggle runs Python 3.13,** which those pins exclude, so the notebook runs the code from source (`PYTHONPATH=src`) instead of installing it. This was rehearsed on Python 3.13 in Docker.
- **The image serves with ONNX Runtime only:** no PyTorch, pandas or uv. It is 544 MB, of which 74 MB is SymPy, a dependency ONNX Runtime declares but does not need for inference.
- **MLflow:** browse a run's records with `uvx --from mlflow==3.17.1 mlflow server --backend-store-uri sqlite:///<run>/mlflow.db`.

## Security

- **PyTorch 2.2.2, the Intel-Mac pin, has known advisories,** including code execution through `torch.load` of a malicious file. No newer PyTorch exists for that platform.
- **Why the risk is contained:**
  - It is used only for local development on the author's machine.
  - It only loads checkpoints from this project's own training runs, with `weights_only=True`; `make export` also checks the selected checkpoint's SHA-256.
  - It is never part of the served image.
- **CI audits** the dependencies it actually installs (Linux, current PyTorch) with `pip-audit`.
- **The API** refuses a model file whose SHA-256 differs from `models/model.json`. It checks image size and aspect ratio from the header before decoding, and never echoes decoder error text.
- **The demo page** loads nothing from third parties. A Content-Security-Policy limits it to its own files, plus Hugging Face's own download domains, to which the Space redirects large files. The page checks the SHA-256 of the model and of all three WebAssembly files (ONNX Runtime and the two image decoders) before using them; those hashes are part of the page's code.

## Limitations

- **The photos are studio-like:** one object, a white background, good light. Real bins, mixed items and clutter were never tested, and accuracy there would likely be much lower. The 68% on the least familiar test photos is a hint of this.
- **The data is small.** The test set has 366 photos in 249 groups, so accuracy is uncertain by about ±3 points. Trash has 19 test photos.
- **The split guarantee stops at 0.86 similarity.** Below it, some test photos may still show an object also photographed in training (46 test photos are at 0.85–0.86), which would make the score slightly optimistic.
- **Plastic in the test set is one family of look-alike photos,** so its score describes plastic in general less well than the other classes' scores do.
- **Validation chose the architecture, the run and the stopping epoch.** The test set was used once. Its intervals were later recomputed from the recorded predictions after a bootstrap bug was fixed; the model was not run on the test photos again.
- **The demo's examples were chosen:** the first validation photo per class that the model gets right. They show what it handles, not its error rate.

## Next steps

- Collect or source photos of real bins and mixed scenes, to evaluate outside the studio setting.
- Temperature-scale the probabilities on validation, so that the displayed confidence means what it says.
- Try longer training: five of six runs had their best epoch near the end, although validation loss was already flat, so the gain may be small. Keep selection on validation.
- Quantise the ONNX model for faster CPU inference, and drop SymPy from the image.

## License

Code: [MIT](LICENSE). Data: TrashNet, MIT, downloaded by the pipeline; seven of its photos are included (the demo examples and `tests/example.jpg`). The model release includes the ONNX file and the PyTorch checkpoint it was exported from.
