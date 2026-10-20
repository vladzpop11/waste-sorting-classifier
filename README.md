# Waste Sorting Classifier

Classify photos of waste into six recycling categories with transfer learning: trained on a free Kaggle GPU, exported to ONNX and served from Docker.

**Status:** data, split, baselines and training code are done and tested; GPU training, export, API and demo are next.

## Data

[TrashNet](https://github.com/garythung/trashnet) (Thung & Yang, 2016), MIT license: 2,527 photos of single objects on a white background, in six classes (cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137). `make data` downloads the archive from [Hugging Face](https://huggingface.co/datasets/garythung/trashnet), checks its SHA-256 and the published class counts.

Two problems in the data, found before any modelling:

1. **The same photo under two labels.** Three photos appear twice, byte for byte, filed under different classes (glass/metal, glass/plastic, glass/plastic). Their true label is unknowable, so all six files are dropped: 2,521 photos remain.
2. **The same object photographed several times.** Many objects appear in several photos, from different angles. A random split would put photos of one object in both training and test, and the test score would partly measure recognising familiar objects.
   - Photos are grouped by the similarity of their ImageNet ResNet18 features: average-linkage clustering within each class, at cosine similarity 0.86. The threshold was chosen by looking at pairs: down to about 0.88 they are clearly the same object, and below 0.86 simple linking chains unrelated photos together.
   - 727 photos (29%) fall into multi-photo groups; the largest has 18.
   - Groups are kept whole when splitting (StratifiedGroupKFold): **1,800 train, 360 validation, 361 test**, with each class at about 14% in validation and test.
   - The split is committed in [splits/split.csv](splits/split.csv), and every image is checked against its recorded SHA-256 when it is loaded.

## Baselines (validation split)

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Majority class | 23.3% | 0.063 |
| Linear probe: frozen ImageNet ResNet18 features + logistic regression | 77.8% | 0.714 |

Fine-tuning has to beat the linear probe to be worth a GPU ([reports/baselines.json](reports/baselines.json)).

## How to run

Requires Python 3.12 via [uv](https://docs.astral.sh/uv/) and `make`. `make help` lists all targets.

```bash
make setup data      # dependencies, then the verified dataset
make check           # Ruff and pytest
make baseline        # the table above
make smoke-train     # one CPU epoch on a small subset, to catch errors cheaply
```

Full training runs on a GPU: [kaggle/train.ipynb](kaggle/train.ipynb) clones this repository and runs the same `train` module for two candidate architectures (ResNet18 and EfficientNet-B0).

- **Choosing between them, on validation only:** the higher macro F1 wins; within 0.01, the faster model wins.
- **Browsing a run's MLflow records** (no extra install):

  ```bash
  uvx --from mlflow==3.17.1 mlflow server --backend-store-uri sqlite:///artifacts/run/resnet18/mlflow.db --port 5000
  ```

## Platform notes

- **Intel Macs** stop at PyTorch 2.2.2 (which needs NumPy 1.x) and ONNX Runtime 1.23.2, so the project pins those there and supports Python 3.12 only. Everywhere else gets current versions; Linux uses PyTorch's CPU-only build.
- **Kaggle runs Python 3.13,** which those Intel-Mac pins exclude, so the notebook runs the code from source (`PYTHONPATH=src`) instead of installing the package. That path is rehearsed on Python 3.13 in Docker.
- **Models trained on Kaggle are exported to ONNX on the Mac:** a checkpoint saved by PyTorch 2.14 loads into 2.2.2 (tested with `weights_only=True`), and exporting with new PyTorch would need extra packages.
- **Serving needs no PyTorch:** the model will be served through ONNX Runtime.

## Security

- **PyTorch 2.2.2, the Intel-Mac pin, has known advisories,** including code execution through `torch.load` of a malicious file. No newer PyTorch exists for that platform.
- **Why the risk is contained:**
  - It is used only for local development on the author's machine.
  - It only ever loads model files produced by this project's own training runs.
  - It is never part of the served image, which uses ONNX Runtime.
- **CI audits** the dependencies it actually installs (Linux, current PyTorch) with `pip-audit`.

## License

Code: [MIT](LICENSE). Data: TrashNet, MIT, downloaded by the pipeline and not included in this repository.
