"""Run track B in order: tune, select, test, then pollution."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tensorflow import keras

from src.data_load import IMAGE_SIZE, load_images
from src.degradation import pollute_train
from src.model_b import build_model_b

RESULTS = Path("results")
FINALIST = RESULTS / "finalistes_B.md"
VARIANTS = ("direct", "padding")
SEEDS = (42, 43, 44)
POLLUTION_LEVELS = (10, 20, 30)
METHOD = "flou"
MAX_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def run_folder(resize, seed, percent=0):
    condition = "clean" if percent == 0 else f"{METHOD}{percent}"
    return RESULTS / f"B_mobilenetv2_frozen_{resize}_{condition}_seed{seed}"


def classification_metrics(y_true, predictions, class_names):
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1_macro": float(f1_score(
            y_true, predictions, labels=labels, average="macro", zero_division=0
        )),
        "class_names": list(class_names),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
    }


def train(resize, seed, percent, train_data, validation_data):
    folder = run_folder(resize, seed, percent)
    if folder.is_dir():
        required = ("weights.weights.h5", "train_loss.csv", "valid_loss.csv", "config.yaml")
        if all((folder / name).is_file() for name in required):
            print(f"Entraînement déjà sauvegardé : {folder}")
            return
        raise RuntimeError(f"Dossier incomplet : {folder}")

    X_train, y_train, _ = train_data
    X_val, y_val, class_names = validation_data
    if percent:
        X_train = pollute_train(X_train, METHOD, percent / 100, seed)

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = build_model_b()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    start = perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=2,
    )
    elapsed = perf_counter() - start
    val_loss, _ = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    val_scores = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    val_metrics = classification_metrics(y_val, val_scores.argmax(axis=1), class_names)

    config = {
        "model": "MobileNetV2 frozen, ImageNet, softmax",
        "resize": resize, "image_size": IMAGE_SIZE, "seed": seed,
        "max_epochs": MAX_EPOCHS, "epochs": len(history.history["loss"]),
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "patience": 3,
        "pollution_method": METHOD if percent else None,
        "pollution_percent": percent, "pollution_seed": seed if percent else None,
        "total_params": int(model.count_params()),
        "trainable_params": sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
        "training_seconds": elapsed,
        "val_loss": float(val_loss),
        "val_accuracy": val_metrics["accuracy"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_confusion_matrix": val_metrics["confusion_matrix"],
        "class_names": list(class_names),
    }
    with TemporaryDirectory(dir=RESULTS) as temporary:
        staged = Path(temporary) / "run"
        staged.mkdir()
        pd.DataFrame({"loss": history.history["loss"]}).to_csv(staged / "train_loss.csv", index=False)
        pd.DataFrame({"loss": history.history["val_loss"]}).to_csv(staged / "valid_loss.csv", index=False)
        model.save_weights(str(staged / "weights.weights.h5"))
        (staged / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        staged.rename(folder)


def read_config(folder):
    return yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))


def evaluate(folder, test_data):
    output = folder / "test_output.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))

    X_test, y_test, class_names = test_data
    keras.backend.clear_session()
    model = build_model_b(weights=None)
    model.load_weights(str(folder / "weights.weights.h5"))
    scores = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    metrics = classification_metrics(y_test, scores.argmax(axis=1), class_names)
    output.write_text(json.dumps(metrics, indent=4) + "\n", encoding="utf-8")

    return metrics


def select():
    losses = {}
    for resize in VARIANTS:
        losses[resize] = np.array([
            read_config(run_folder(resize, seed))["val_loss"] for seed in SEEDS
        ])
    difference = losses["direct"] - losses["padding"]
    # Prefer direct when the paired difference is within its observed variability.
    resize = "padding" if difference.mean() > difference.std(ddof=1) else "direct"
    lines = [
        "# Finaliste de la piste B", "",
        f"Redimensionnement retenu : `{resize}`", "",
        f"Configurations essayées : {len(VARIANTS)}. Seeds : {', '.join(map(str, SEEDS))}.",
        "", "Loss de validation (moyenne ± écart-type) :",
    ]
    for variant in VARIANTS:
        lines.append(f"- {variant} : {losses[variant].mean():.6f} ± {losses[variant].std(ddof=1):.6f}")
    lines += ["", "Critère : loss de validation, avec préférence pour le direct si son",
              "écart moyen avec les marges est inférieur ou égal à l’écart-type",
              "des différences par seed.", "", "Runs retenus :"]
    lines.extend(f"- `{run_folder(resize, seed).name}`" for seed in SEEDS)
    with FINALIST.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def report(resize, percents, test_data):
    rows = []
    for percent in percents:
        for seed in SEEDS:
            folder = run_folder(resize, seed, percent)
            metrics = evaluate(folder, test_data)
            rows.append({
                "pollution_percent": percent, "seed": seed,
                "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
            })
    grouped = pd.DataFrame(rows).groupby("pollution_percent")
    summary = grouped[["accuracy", "f1_macro"]].agg(["mean", "std"])
    print(summary.to_string())
    if len(percents) > 1:
        summary.columns = ["_".join(column) for column in summary.columns]
        summary.to_csv(RESULTS / "B_pollution_summary.csv")


def load_split(split, resize):
    return load_images(split, with_padding=resize == "padding")


def train_runs(resize, percentages):
    train_data = load_split("train", resize)
    validation_data = load_split("validation", resize)
    for percent in percentages:
        for seed in SEEDS:
            train(resize, seed, percent, train_data, validation_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tune", "select", "test", "pollution"])
    stage = parser.parse_args().stage
    RESULTS.mkdir(exist_ok=True)

    if stage == "tune":
        for resize in VARIANTS:
            train_runs(resize, (0,))
        return
    if stage == "select":
        select()
        print(FINALIST)
        return

    if not FINALIST.is_file():
        parser.error("Lance d’abord select pour figer le finaliste sur la validation.")
    declared = FINALIST.read_text(encoding="utf-8")
    resize = next((v for v in VARIANTS if f"`{run_folder(v, SEEDS[0]).name}`" in declared), None)
    if resize is None:
        parser.error(f"Finaliste illisible dans {FINALIST}.")

    test_data = load_split("test", resize)
    if stage == "test":
        report(resize, (0,), test_data)
        return

    if not all((run_folder(resize, seed) / "test_output.json").is_file() for seed in SEEDS):
        parser.error("Lance d’abord test pour mesurer le finaliste propre.")
    train_runs(resize, POLLUTION_LEVELS)
    report(resize, (0, *POLLUTION_LEVELS), test_data)


if __name__ == "__main__":
    main()
