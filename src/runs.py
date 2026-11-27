"""Run naming, resume helpers and readers for results/.

This module does NOT import TensorFlow : it stays light, so the launcher can use it
without loading the deep learning libraries.
"""
import json
from pathlib import Path

import pandas as pd
import yaml

RESULTS_DIR = Path("results")


def run_name(
    unfreeze_last_n_layers=0,
    lr_head=1e-3,
    lr_finetune=1e-5,
    pollution_method=None,
    pollution_fraction=0.0,
    seed=42,
    tag="",
    **_ignored,
):
    """Deterministic folder name : the same config always gives the same name.

    Extra keys are ignored, so a whole run config can be passed with **config.
    Use `tag` to tell apart two runs that differ only by a parameter absent from the name.
    """
    phase = "frozen" if unfreeze_last_n_layers == 0 else f"unfreeze{unfreeze_last_n_layers}layers"
    lr = lr_head if unfreeze_last_n_layers == 0 else lr_finetune
    pollution_percent = int(round(pollution_fraction * 100))
    pollution = "clean" if pollution_percent == 0 else f"{pollution_method}{pollution_percent}"
    parts = ["C_mobilenetv2", phase, f"lr{lr:g}", pollution]
    if tag:
        parts.append(tag)
    parts.append(f"seed{seed}")
    return "_".join(parts)


def run_dir(config):
    return RESULTS_DIR / run_name(**config)


def is_done(config):
    """A run is finished when its final folder exists (it is renamed from .partial at the very end)."""
    return run_dir(config).is_dir()


def has_test(config):
    return (run_dir(config) / "test_output.json").exists()


def valid_disagreements(run_a, run_b):
    """Compare two runs image by image on the validation set (names of folders in results/).

    Two variants that disagree on only a few images are not significantly different.
    """
    a = pd.read_csv(RESULTS_DIR / run_a / "valid_predictions.csv")
    b = pd.read_csv(RESULTS_DIR / run_b / "valid_predictions.csv")
    a_ok = a["y_pred"] == a["y_true"]
    b_ok = b["y_pred"] == b["y_true"]
    return {
        "n_images": len(a),
        "n_different_predictions": int((a["y_pred"] != b["y_pred"]).sum()),
        "only_a_correct": int((a_ok & ~b_ok).sum()),
        "only_b_correct": int((~a_ok & b_ok).sum()),
        "both_wrong": int((~a_ok & ~b_ok).sum()),
    }


def summarize_runs(pattern="*", average_seeds=True):
    """Validation metrics of the finished runs (the test is never shown here).

    average_seeds=True : one line per configuration, mean and std over the seeds.
    """
    rows = []
    for folder in sorted(RESULTS_DIR.glob(pattern)):
        if not folder.is_dir() or folder.name.endswith(".partial"):
            continue
        config_file, valid_file = folder / "config.yaml", folder / "valid_output.json"
        if not (config_file.exists() and valid_file.exists()):
            continue
        config = yaml.safe_load(config_file.read_text())
        valid = json.loads(valid_file.read_text())
        rows.append(
            {
                "run": folder.name,
                "unfreeze_last_n_layers": config["unfreeze_last_n_layers"],
                "lr_head": config["lr_head"],
                "lr_finetune": config["lr_finetune"],
                "pollution_percent": int(round(config["pollution_fraction"] * 100)),
                "tag": config["tag"],
                "seed": config["seed"],
                "val_loss": valid["loss"],
                "val_accuracy": valid["accuracy"],
                "val_f1_macro": valid["f1_macro"],
                "trainable_params": config["trainable_params"],
                "training_time_s": config["training_time_s"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty or not average_seeds:
        return df
    keys = ["unfreeze_last_n_layers", "lr_head", "lr_finetune", "pollution_percent", "tag"]
    metrics = ["val_loss", "val_accuracy", "val_f1_macro"]
    grouped = df.groupby(keys)
    out = grouped[metrics].agg(["mean", "std"])
    out["n_seeds"] = grouped["seed"].count()
    return out.sort_values(("val_loss", "mean"))
