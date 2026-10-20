"""Fine-tune a pre-trained CNN on the training split, early-stopping on validation loss.

Runs on a GPU when available (Kaggle) and on the CPU otherwise. Every epoch is logged to
MLflow (a SQLite store in the output directory). Writes to --output-dir:
  model.pt          best weights (lowest validation loss) + class names + architecture
  metrics.json      validation metrics of the best epoch, run settings, versions
  history.json      per-epoch losses and validation metrics
  confusion_val.png validation confusion matrix of the best epoch
  mlflow.db, mlartifacts/  the MLflow tracking store (SQLite) and artifacts
The test split is not used here.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from waste_sorting_classifier.config import load_config, project_path
from waste_sorting_classifier.dataset import WasteImages
from waste_sorting_classifier.model import build_model
from waste_sorting_classifier.split import load_split

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_per_class(rows, n: int | None, classes: list[str] | None, seed: int):
    if classes:
        rows = rows[rows["label"].isin(classes)]
    if n:
        # Explicit concat: pandas 3 drops the grouping column inside groupby().apply().
        rows = pd.concat(
            [g.sample(min(n, len(g)), random_state=seed) for _, g in rows.groupby("label")]
        )
    return rows.reset_index(drop=True)


def evaluate(model, loader, device, loss_fn) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total, logits_all, targets_all = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += loss_fn(logits, y).item() * len(y)
            logits_all.append(logits.float().cpu())
            targets_all.append(y.cpu())
    logits = torch.cat(logits_all)
    return total / len(logits), logits.softmax(1).numpy(), torch.cat(targets_all).numpy()


def save_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    ax.set(xlabel="predicted", ylabel="true", title="Validation confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=Path, default=project_path("artifacts/run"))
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--arch", default=None, help="override train.arch (resnet18, efficientnet_b0)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit-per-class", type=int, default=None, help="smoke runs only")
    p.add_argument("--classes", nargs="*", default=None, help="smoke runs only")
    p.add_argument("--no-pretrained", action="store_true", help="tests only")
    p.add_argument("--no-verify", action="store_true", help="skip image SHA-256 checks")
    args = p.parse_args(argv)

    cfg = load_config()
    tc = dict(cfg["train"])
    if args.arch:
        tc["arch"] = args.arch
    epochs = args.epochs or tc["epochs"]
    batch_size = args.batch_size or tc["batch_size"]
    seed_everything(cfg["seed"])
    device = pick_device(args.device)

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = tc["num_workers"]
        if num_workers == "auto":
            num_workers = 2 if device.type == "cuda" else 0

    images_dir = args.images_dir or project_path(cfg["data"]["raw_dir"]) / "images"
    split = load_split(cfg, verify=not args.no_verify, images_dir=images_dir)
    classes = args.classes or cfg["data"]["classes"]
    rows = {
        name: limit_per_class(
            split[split["split"] == name], args.limit_per_class, args.classes, cfg["seed"]
        )  # fmt: skip
        for name in ["train", "val"]
    }
    loaders = {
        name: torch.utils.data.DataLoader(
            WasteImages(rows[name], images_dir, classes, train=name == "train"),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )
        for name in ["train", "val"]
    }

    model = build_model(tc["arch"], len(classes), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    # torch.amp.GradScaler exists from PyTorch 2.3; the Intel-Mac build (2.2.2) has only the
    # older torch.cuda.amp one. Mixed precision is used on GPUs only.
    use_amp = device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    import mlflow

    # A SQLite tracking store and an artifact folder inside the output directory, so a run
    # (e.g. downloaded from Kaggle) is self-contained: mlflow ui --backend-store-uri ...
    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.db').resolve()}")
    if mlflow.get_experiment_by_name("waste-sorting") is None:
        mlflow.create_experiment(
            "waste-sorting", artifact_location=(out / "mlartifacts").resolve().as_uri()
        )
    mlflow.set_experiment("waste-sorting")
    params = {
        **tc,
        "num_workers": num_workers,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device.type,
        "train_images": len(rows["train"]),
        "val_images": len(rows["val"]),
        "pretrained": not args.no_pretrained,
        "classes": ",".join(classes),
    }

    history, best, stale = [], None, 0
    with mlflow.start_run():
        mlflow.log_params(params)
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            model.train()
            seen, train_loss = 0, 0.0
            for x, y in loaders["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = loss_fn(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * len(y)
                seen += len(y)
            scheduler.step()

            val_loss, proba, targets = evaluate(model, loaders["val"], device, loss_fn)
            pred = proba.argmax(1)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / seen,
                "val_loss": val_loss,
                "val_accuracy": accuracy_score(targets, pred),
                "val_macro_f1": f1_score(targets, pred, average="macro", zero_division=0),
                "seconds": time.perf_counter() - start,
            }
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in row.items()), flush=True)  # fmt: skip

            if best is None or val_loss < best["val_loss"]:
                best, stale = {**row, "confusion": confusion_matrix(targets, pred).tolist()}, 0
                torch.save(
                    {"state_dict": model.state_dict(), "classes": classes, "arch": tc["arch"]},
                    out / "model.pt",
                )
            else:
                stale += 1
                if stale >= tc["patience"]:
                    print(f"early stopping after epoch {epoch}")
                    break

        save_confusion(np.array(best["confusion"]), classes, out / "confusion_val.png")
        metrics = {
            "best_epoch": best["epoch"],
            "val_loss": best["val_loss"],
            "val_accuracy": best["val_accuracy"],
            "val_macro_f1": best["val_macro_f1"],
            "val_confusion": best["confusion"],
            "classes": classes,
            "params": params,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            },  # fmt: skip
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        mlflow.log_metrics({"best_val_loss": best["val_loss"],
                            "best_val_macro_f1": best["val_macro_f1"]})  # fmt: skip
        mlflow.log_artifact(str(out / "confusion_val.png"))
    print(f"best epoch {best['epoch']}: val accuracy {best['val_accuracy']:.4f}, "
          f"macro F1 {best['val_macro_f1']:.4f} -> {out}")  # fmt: skip
    return metrics


if __name__ == "__main__":
    main()
