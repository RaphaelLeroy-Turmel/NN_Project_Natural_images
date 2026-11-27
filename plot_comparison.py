"""Track C : comparison figures of the finalists. Run it after the `test` and `pollution` steps.

    python plot_comparison.py

It only reads results/ : nothing is retrained and no image is loaded. It writes in
results/figures/ :

    1_tuning_validation_<method>_<NNN>.png   validation loss of every tuned config (mean +- std over the seeds)
    2_pollution_curves_<method>_<NNN>.png    test loss and test F1 macro vs pollution level, one line per finalist
    3_learning_curves_<method>_<NNN>.png     train / validation loss per epoch (finalists, clean, seed 42)
    4_confusion_test_<method>_<NNN>.png      test errors summed over the 3 seeds (finalists, clean)
    test_summary_<method>_<NNN>.csv          the numbers behind figure 2

<method> is POLLUTION_METHOD of run_finetuning.py. <NNN> is a number that grows at every
execution (001, 002, ...) : an execution never overwrites the files of a previous one.

and prints the tables used for the analysis : copy the whole console output.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from run_finetuning import FINALISTS, POLLUTION_LEVELS, POLLUTION_METHOD, SEEDS
from src.runs import RESULTS_DIR, run_name

FIG_DIR = RESULTS_DIR / "figures"

# Colors : one fixed color per category (validated for color blindness), text in neutral inks
COLOR_FROZEN = "#257de0"     # blue
COLOR_FINETUNE = "#f47230"   # orange
INK, INK_2, GRID = "#191919", "#666666", "#e7e7e7"
MARKER_FROZEN, MARKER_FINETUNE = "o", "s"  # second encoding : the shape, not only the color

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c2c2c2", "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def label(unfreeze):
    return "Frozen backbone" if unfreeze == 0 else f"Fine-tuning ({unfreeze} layers)"


def style(unfreeze):
    if unfreeze == 0:
        return dict(color=COLOR_FROZEN, marker=MARKER_FROZEN)
    return dict(color=COLOR_FINETUNE, marker=MARKER_FINETUNE)


def next_index():
    """1 + the highest <NNN> already used in results/figures/ (001 if the folder is empty)."""
    used = [int(m.group(1)) for f in FIG_DIR.iterdir()
            if (m := re.search(r"_(\d{3})\.(png|csv)$", f.name))]
    return max(used, default=0) + 1


def out_path(stem, suffix="png"):
    """File name of this execution : <stem>_<method>_<NNN>.<suffix>."""
    return FIG_DIR / f"{stem}_{POLLUTION_METHOD}_{RUN_INDEX:03d}.{suffix}"


RUN_INDEX = None  # set in main()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- loading


def load_all_runs():
    """One row per finished track C run : config, validation and (if any) test metrics."""
    rows = []
    for folder in sorted(RESULTS_DIR.glob("C_*")):
        if not folder.is_dir() or folder.name.endswith(".partial"):
            continue
        config_file = folder / "config.yaml"
        valid = read_json(folder / "valid_output.json")
        if not config_file.exists() or valid is None:
            continue
        config = yaml.safe_load(config_file.read_text())
        test = read_json(folder / "test_output.json")
        rows.append({
            "run": folder.name,
            "unfreeze": config["unfreeze_last_n_layers"],
            "lr": config["lr_head"] if config["unfreeze_last_n_layers"] == 0 else config["lr_finetune"],
            "pollution_percent": int(round(config["pollution_fraction"] * 100)),
            "seed": config["seed"],
            "val_loss": valid["loss"], "val_accuracy": valid["accuracy"], "val_f1": valid["f1_macro"],
            "test_loss": test["loss"] if test else np.nan,
            "test_accuracy": test["accuracy"] if test else np.nan,
            "test_f1": test["f1_macro"] if test else np.nan,
            "test_errors": (np.sum(test["confusion_matrix"]) - np.trace(test["confusion_matrix"]))
                           if test else np.nan,
            "trainable_params": config["trainable_params"],
            "total_params": config["total_params"],
            "epochs_run": config["epochs_run"],
            "training_time_s": config["training_time_s"],
        })
    return pd.DataFrame(rows)


def finalist_runs(config):
    """Folder names of one finalist : {pollution_percent: [run names, one per seed]}."""
    out = {0: [run_name(**config, seed=s) for s in SEEDS]}
    for level in POLLUTION_LEVELS:
        out[int(round(level * 100))] = [
            run_name(**config, seed=s, pollution_method=POLLUTION_METHOD, pollution_fraction=level)
            for s in SEEDS
        ]
    return out


def mean_std(values):
    values = pd.Series(values, dtype=float).dropna()
    return values.mean(), values.std(), len(values)  # std with ddof=1, like summarize_runs


# ---------------------------------------------------------------- figure 1


def plot_tuning(runs):
    clean = runs[runs["pollution_percent"] == 0]
    grouped = (clean.groupby(["unfreeze", "lr"])["val_loss"]
               .agg(["mean", "std", "count"]).reset_index().sort_values("mean", ascending=False))
    finalist_keys = {(c.get("unfreeze_last_n_layers", 0),
                      c.get("lr_head", 1e-3) if c.get("unfreeze_last_n_layers", 0) == 0
                      else c.get("lr_finetune", 1e-5)) for c in FINALISTS}

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(grouped) + 1.2))
    y = np.arange(len(grouped))
    for yi, row in zip(y, grouped.itertuples()):
        color = style(row.unfreeze)["color"]
        ax.barh(yi, row.mean, height=0.6, color=color)
        if row.count > 1:
            ax.errorbar(row.mean, yi, xerr=row.std, color=INK, capsize=3, linewidth=1)
        tag = "  finalist" if (row.unfreeze, row.lr) in finalist_keys else ""
        ax.text(row.mean + (row.std if row.count > 1 else 0) + grouped["mean"].max() * 0.01, yi,
                f"{row.mean:.4f}  (n={row.count}){tag}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [f"{label(u)}, lr {lr:g}" for u, lr in zip(grouped["unfreeze"], grouped["lr"])])
    ax.set_xlabel("Validation loss (lower is better) — mean ± std over seeds")
    ax.set_xlim(0, grouped["mean"].max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_title("Track C tuning : validation loss of every configuration", loc="left")
    fig.savefig(out_path("1_tuning_validation"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 2


def test_table(runs):
    by_name = runs.set_index("run")
    rows = []
    for config in FINALISTS:
        unfreeze = config.get("unfreeze_last_n_layers", 0)
        for percent, names in finalist_runs(config).items():
            present = by_name.loc[by_name.index.intersection(names)]
            row = {"finalist": label(unfreeze), "unfreeze": unfreeze, "pollution_percent": percent}
            for metric in ("test_loss", "test_accuracy", "test_f1", "test_errors", "val_loss"):
                row[f"{metric}_mean"], row[f"{metric}_std"], row["n_seeds"] = mean_std(present[metric])
            row["training_time_s_mean"] = present["training_time_s"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_pollution(table):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axes[0], "test_loss", "Test loss (lower is better)"),
                              (axes[1], "test_f1", "Test F1 macro (higher is better)")]:
        for unfreeze, group in table.groupby("unfreeze"):
            group = group.sort_values("pollution_percent")
            ax.errorbar(group["pollution_percent"], group[f"{metric}_mean"], yerr=group[f"{metric}_std"],
                        linewidth=2, markersize=8, capsize=3, label=label(unfreeze), **style(unfreeze))
        ax.set_xticks(sorted(table["pollution_percent"].unique()))
        ax.set_xlabel(f"Polluted training images (%) — method : {POLLUTION_METHOD}")
        ax.set_title(title, loc="left")
        ax.set_xlim(-2, max(table["pollution_percent"]) + 2)
        ax.legend()
    fig.suptitle(f"Finalists under training-data pollution (mean ± std over {len(SEEDS)} seed(s))", x=0.01, y=1.04, ha="left",
                 color=INK)
    fig.savefig(out_path("2_pollution_curves"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 3


def plot_learning_curves():
    fig, axes = plt.subplots(1, len(FINALISTS), figsize=(5.5 * len(FINALISTS), 4), sharey=True, squeeze=False)
    for ax, config in zip(axes[0], FINALISTS):
        unfreeze = config.get("unfreeze_last_n_layers", 0)
        folder = RESULTS_DIR / run_name(**config, seed=SEEDS[0])
        train = pd.read_csv(folder / "train_loss.csv")
        valid = pd.read_csv(folder / "valid_loss.csv")
        color = style(unfreeze)["color"]
        ax.plot(train["epoch"], train["loss"], color=color, linewidth=2, linestyle="--", label="train")
        ax.plot(valid["epoch"], valid["loss"], color=color, linewidth=2, label="validation")
        best = valid["loss"].idxmin()
        ax.plot(valid["epoch"][best], valid["loss"][best], "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"best val {valid['loss'][best]:.4f}", (valid["epoch"][best], valid["loss"][best]),
                    xytext=(0, 10), textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch (phase 1 then phase 2 if any)")
        ax.set_title(label(unfreeze), loc="left")
        ax.legend()
    axes[0][0].set_ylabel("Loss (log scale)")
    fig.suptitle(f"Learning curves — clean data, seed {SEEDS[0]}", x=0.01, y=1.04, ha="left", color=INK)
    fig.savefig(out_path("3_learning_curves"))
    plt.close(fig)


# ---------------------------------------------------------------- figure 4


def summed_confusion(config):
    total, classes = None, None
    for name in finalist_runs(config)[0]:
        test = read_json(RESULTS_DIR / name / "test_output.json")
        if test is None:
            continue
        cm = np.array(test["confusion_matrix"])
        total = cm if total is None else total + cm
        classes = test["classes"]
    return total, classes


def plot_confusions():
    fig, axes = plt.subplots(1, len(FINALISTS), figsize=(6 * len(FINALISTS), 5.2), squeeze=False)
    matrices = [summed_confusion(c) for c in FINALISTS]
    vmax = max(int((cm - np.diag(np.diag(cm))).max()) for cm, _ in matrices if cm is not None) or 1
    for ax, config, (cm, classes) in zip(axes[0], FINALISTS, matrices):
        if cm is None:
            ax.set_axis_off()
            continue
        errors = cm.astype(float).copy()
        np.fill_diagonal(errors, np.nan)  # the diagonal is shown as text only : the colour is for errors
        ax.imshow(errors, cmap="Blues", vmin=0, vmax=vmax)
        n = len(classes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, cm[i, j], ha="center", va="center", color=INK_2, fontsize=8)
                elif cm[i, j] > 0:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, fontweight="bold",
                            color="white" if cm[i, j] > vmax / 2 else INK)
        ax.set_xticks(range(n), classes, rotation=45, ha="right")
        ax.set_yticks(range(n), classes)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.grid(False)
        unfreeze = config.get("unfreeze_last_n_layers", 0)
        ax.set_title(f"{label(unfreeze)} — {int(errors[~np.isnan(errors)].sum())} errors", loc="left")
    fig.suptitle(f"Test confusion matrices, summed over {len(SEEDS)} seeds (colour = errors only)",
                 x=0.01, y=1.04, ha="left", color=INK)
    fig.savefig(out_path("4_confusion_test"))
    plt.close(fig)
    return matrices


# ---------------------------------------------------------------- console report


def print_report(runs, table, matrices):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n=== 1. Test metrics of the finalists (mean ± std over seeds) ===")
    view = table[["finalist", "pollution_percent", "n_seeds",
                  "test_loss_mean", "test_loss_std", "test_accuracy_mean", "test_accuracy_std",
                  "test_f1_mean", "test_f1_std", "test_errors_mean", "val_loss_mean"]]
    print(view.round(5).to_string(index=False))

    print("\n=== 2. Cost of the finalists (clean runs) ===")
    for config in FINALISTS:
        unfreeze = config.get("unfreeze_last_n_layers", 0)
        clean = runs[runs["run"].isin(finalist_runs(config)[0])]
        print(f"{label(unfreeze):28s} trainable params {int(clean['trainable_params'].iloc[0]):>9,d} / "
              f"{int(clean['total_params'].iloc[0]):,d}   epochs run {clean['epochs_run'].mean():.1f}   "
              f"training time {clean['training_time_s'].mean():.0f} s (mean)")

    print("\n=== 3. Test errors on clean data, summed over seeds (true -> predicted : count) ===")
    for config, (cm, classes) in zip(FINALISTS, matrices):
        unfreeze = config.get("unfreeze_last_n_layers", 0)
        if cm is None:
            print(f"{label(unfreeze)} : no test_output.json found")
            continue
        pairs = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
                 for j in range(len(classes)) if i != j and cm[i, j] > 0]
        pairs.sort(reverse=True)
        print(f"{label(unfreeze)} :", ", ".join(f"{t} -> {p} : {c}" for c, t, p in pairs) or "no error")

    missing = table[table["n_seeds"] < len(SEEDS)]
    if not missing.empty:
        print("\nWARNING : some finalist runs have no test result :")
        print(missing[["finalist", "pollution_percent", "n_seeds"]].to_string(index=False))
    print(f"\nFigures written in {FIG_DIR} with the suffix _{POLLUTION_METHOD}_{RUN_INDEX:03d}")


def main():
    if not FINALISTS:
        raise SystemExit("FINALISTS is empty in run_finetuning.py : nothing to compare.")
    global RUN_INDEX
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_INDEX = next_index()
    runs = load_all_runs()
    table = test_table(runs)
    table.to_csv(out_path("test_summary", "csv"), index=False)

    plot_tuning(runs)
    plot_pollution(table)
    plot_learning_curves()
    matrices = plot_confusions()
    print_report(runs, table, matrices)


if __name__ == "__main__":
    main()