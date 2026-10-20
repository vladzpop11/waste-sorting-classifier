"""Apply the model-selection rule in configs/config.yaml to a directory of training runs.

Reads RUNS_DIR/<arch>/seed<seed>/metrics.json for every architecture and seed, and writes
reports/model_selection.json: every run's validation results, each architecture's mean and
standard deviation, the decision, and the chosen checkpoint with its SHA-256. Only the
validation split is used.
"""

import json
import statistics
import sys
from pathlib import Path

from waste_sorting_classifier.artifacts import sha256, write_json
from waste_sorting_classifier.config import load_config, project_path


def decide(means: dict[str, float], margin: float, faster: str) -> str:
    best = max(means, key=means.get)
    if any(a != best and means[best] - m <= margin for a, m in means.items()):
        return faster
    return best


def score(metrics: dict, rule: dict) -> float:
    """The validation score the rule compares: macro F1, or (version 4) its mean over the
    data sources, so that each source counts equally."""
    if rule.get("score") == "mean_over_sources":
        return statistics.mean(metrics[f"val_macro_f1_{s}"] for s in rule["sources"])
    return metrics["val_macro_f1"]


def main() -> None:
    cfg = load_config()
    rule = cfg["selection"]
    runs_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else project_path(rule["runs_dir"])
    runs = {}
    for arch in rule["archs"]:
        for seed in rule["seeds"]:
            run = runs_dir / arch / f"seed{seed}"
            m = json.loads((run / "metrics.json").read_text())
            if m["params"]["seed"] != seed or m["params"]["arch"] != arch:
                raise SystemExit(f"{run}: metrics.json describes another run")
            runs[f"{arch}/seed{seed}"] = {
                "arch": arch,
                "seed": seed,
                "best_epoch": m["best_epoch"],
                "val_loss": m["val_loss"],
                "val_accuracy": m["val_accuracy"],
                "val_macro_f1": m["val_macro_f1"],
                **{k: v for k, v in m.items() if k.startswith(("val_macro_f1_", "val_accuracy_"))},
                "score": score(m, rule),
                "checkpoint_sha256": sha256(run / "model.pt"),
                "versions": m["versions"],
            }
    per_arch = {}
    for arch in rule["archs"]:
        f1 = [r["score"] for r in runs.values() if r["arch"] == arch]
        acc = [r["val_accuracy"] for r in runs.values() if r["arch"] == arch]
        per_arch[arch] = {
            "mean_score": statistics.mean(f1),
            "sd_score": statistics.stdev(f1) if len(f1) > 1 else 0.0,
            "mean_val_accuracy": statistics.mean(acc),
        }
    means = {a: v["mean_score"] for a, v in per_arch.items()}
    chosen_arch = decide(means, rule["margin"], rule["faster"])
    chosen = min(
        (k for k, r in runs.items() if r["arch"] == chosen_arch), key=lambda k: runs[k]["val_loss"]
    )
    commit = (
        (runs_dir / "commit.txt").read_text().strip()
        if (runs_dir / "commit.txt").exists()
        else None
    )
    record = {
        "rule": rule,
        "commit": commit,
        "runs": runs,
        "per_arch": per_arch,
        "chosen_arch": chosen_arch,
        "chosen_run": chosen,
        "chosen_checkpoint": str((runs_dir / chosen / "model.pt").relative_to(project_path(""))),
        "chosen_checkpoint_sha256": runs[chosen]["checkpoint_sha256"],
    }
    write_json(project_path(cfg["paths"]["reports_dir"]) / "model_selection.json", record)
    for arch, v in per_arch.items():
        f1, sd, acc = v["mean_score"], v["sd_score"], v["mean_val_accuracy"]
        print(f"{arch:16s} mean validation score {f1:.4f} (sd {sd:.4f}), mean accuracy {acc:.4f}")
    print(f"chosen: {chosen_arch}, run {chosen} (val loss {runs[chosen]['val_loss']:.4f})")


if __name__ == "__main__":
    main()
