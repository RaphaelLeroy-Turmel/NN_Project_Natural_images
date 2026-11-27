import json
import time
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

from src.data_load import IMAGE_SIZE
from src.degradation import pollute_train
from src.evaluation import save_results
from src.model1 import ACTIVATION, ARCHITECTURE, CONV_FILTERS, DENSE_UNITS, build_model

RESULTS_DIR = Path("results")
WEIGHTS_FILE = "model.weights.h5"
SEEDS = (42, 43, 44)
TUNING_SEED = 42
MAX_EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 3
POLLUTION_METHOD = "flou"
POLLUTION_PERCENTS = (0, 10, 20, 30)


def category_name(augment):
    return "aug" if augment else "noaug"


def config_key(augment, lr, dropout):
    return f"A_cnn4blocks_{category_name(augment)}_lr{lr:g}_do{dropout:g}"


def run_name(key, seed, pollution_percent=0):
    pollution = "clean" if pollution_percent == 0 else f"{POLLUTION_METHOD}{pollution_percent}"
    return f"{key}_{pollution}_seed{seed}"


def is_done(name):
    return (RESULTS_DIR / name / "config.yaml").is_file()


def load_config(name):
    with (RESULTS_DIR / name / "config.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model(name, n_classes):
    config = load_config(name)
    model = build_model(n_classes, augment=config["augmentation"], dropout_rate=config["dropout_rate"])
    model.load_weights(str(RESULTS_DIR / name / WEIGHTS_FILE))
    return model


def train_run(data, augment, lr, dropout, seed, pollution_percent=0, max_epochs=MAX_EPOCHS):
    key = config_key(augment, lr, dropout)
    name = run_name(key, seed, pollution_percent)
    if is_done(name):
        print(f"{name} : déjà terminé, non relancé")
        return name

    keras.utils.set_random_seed(seed)
    X_train = data["X_train"]
    if pollution_percent > 0:
        X_train = pollute_train(X_train, POLLUTION_METHOD, pollution_percent / 100, seed)

    model = build_model(len(data["class_names"]), augment=augment, dropout_rate=dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    print(f"--- {name}")
    start = time.perf_counter()
    history = model.fit(
        X_train, data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=max_epochs, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=2,
    )
    training_seconds = time.perf_counter() - start

    val_predictions = model.predict(data["X_val"], verbose=0).argmax(axis=1)
    best_epoch = int(np.argmin(history.history["val_loss"]))

    config = {
        "run_name": name,
        "config_key": key,
        "track": "A",
        "model": "cnn_from_scratch",
        "category": category_name(augment),
        "architecture": ARCHITECTURE,
        "conv_filters": list(CONV_FILTERS),
        "activation": ACTIVATION,
        "dense_units": DENSE_UNITS,
        "dropout_rate": float(dropout),
        "augmentation": bool(augment),
        "normalization": "Rescaling(1/255) inside the model",
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "pretrained": False,
        "optimizer": "adam",
        "learning_rate": float(lr),
        "batch_size": BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping": {"monitor": "val_loss", "patience": PATIENCE, "restore_best_weights": True},
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best_epoch + 1,
        "n_parameters": int(model.count_params()),
        "n_trainable_parameters": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "seed": int(seed),
        "pollution_method": POLLUTION_METHOD if pollution_percent > 0 else None,
        "pollution_percent": int(pollution_percent),
        "training_seconds": round(training_seconds, 1),
        "validation": {
            "loss": float(history.history["val_loss"][best_epoch]),
            "accuracy": float(accuracy_score(data["y_val"], val_predictions)),
            "f1_macro": float(f1_score(data["y_val"], val_predictions, average="macro")),
        },
    }

    folder = RESULTS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(folder / WEIGHTS_FILE))
    pd.DataFrame({"loss": history.history["loss"]}).to_csv(folder / "train_loss.csv", index=False)
    pd.DataFrame({"loss": history.history["val_loss"]}).to_csv(folder / "valid_loss.csv", index=False)
    with (folder / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    print(f"{name} : val_loss={config['validation']['loss']:.4f}  "
          f"val_f1={config['validation']['f1_macro']:.4f}  ({training_seconds:.0f}s)")
    return name


def evaluate_on_test(data, name):
    output = RESULTS_DIR / name / "test_output.json"
    if output.is_file():
        with output.open(encoding="utf-8") as file:
            return json.load(file)

    model = load_model(name, len(data["class_names"]))
    scores = model.predict(data["X_test"], verbose=0)
    history = SimpleNamespace(history={
        "loss": pd.read_csv(RESULTS_DIR / name / "train_loss.csv")["loss"].tolist(),
        "val_loss": pd.read_csv(RESULTS_DIR / name / "valid_loss.csv")["loss"].tolist(),
    })
    return save_results(name, model, history, data["y_test"], scores,
                        data["class_names"], load_config(name))


def runs_table(names, with_test=False):
    rows = []
    for name in names:
        config = load_config(name)
        row = {
            "run": name,
            "config_key": config["config_key"],
            "category": config["category"],
            "learning_rate": config["learning_rate"],
            "dropout_rate": config["dropout_rate"],
            "seed": config["seed"],
            "pollution_percent": config["pollution_percent"],
            "val_loss": config["validation"]["loss"],
            "val_f1_macro": config["validation"]["f1_macro"],
            "val_accuracy": config["validation"]["accuracy"],
            "epochs_run": config["epochs_run"],
            "n_parameters": config["n_parameters"],
            "n_trainable_parameters": config["n_trainable_parameters"],
            "training_seconds": config["training_seconds"],
        }
        if with_test:
            with (RESULTS_DIR / name / "test_output.json").open(encoding="utf-8") as file:
                test = json.load(file)
            row["test_accuracy"] = test["accuracy"]
            row["test_f1_macro"] = test["f1_macro"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_finalists(finalists, n_configs_tried, path=RESULTS_DIR / "finalistes_A.md"):
    lines = [
        "# Finalistes de la piste A (CNN from scratch)",
        "",
        f"Désignés le {date.today().isoformat()} sur la loss de validation (moyenne sur les seeds "
        f"{', '.join(str(s) for s in SEEDS)}), avant toute lecture du test.",
        "",
        "| Catégorie | Finaliste | Learning rate | Dropout | Configurations essayées |",
        "|---|---|---|---|---|",
    ]
    for category, finalist in finalists.items():
        lines.append(
            f"| {category} | `{config_key(**finalist)}` | {finalist['lr']:g} | "
            f"{finalist['dropout']:g} | {n_configs_tried[category]} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
