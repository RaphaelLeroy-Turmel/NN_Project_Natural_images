"""Track C : run one experiment (train, evaluate, save everything in results/)."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # pip install pyyaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.degradation import pollute_train
from src.model_finetuning import build_model, set_unfrozen_layers

RESULTS_DIR = Path("results")


def make_run_dir(base_name):
    """Create results/<base_name>_v<n>, never overwriting a previous run."""
    version = 1
    while (RESULTS_DIR / f"{base_name}_v{version}").exists():
        version += 1
    run_dir = RESULTS_DIR / f"{base_name}_v{version}"
    run_dir.mkdir(parents=True)
    return run_dir


def count_params(weights):
    return int(sum(np.prod(tuple(w.shape)) for w in weights))


def run_experiment(
    X_train, y_train, X_valid, y_valid, X_test, y_test, class_names,
    unfreeze_last_n_layers=0,   # 0 = backbone frozen only (like track B)
    lr_head=1e-3,               # phase 1 : train the new head
    lr_finetune=1e-5,           # phase 2 : very low lr on the unfrozen layers
    epochs_head=10,
    epochs_finetune=10,
    batch_size=32,
    pollution_method=None,      # "flou", "pixelisation", "masque" or None
    pollution_fraction=0.0,     # fraction of the TRAIN images that are degraded
    seed=42,
):
    """Train one model, evaluate it on the test set and save everything in results/.

    The X arrays must contain raw uint8 pixels : normalization is done here.
    """
    keras.utils.set_random_seed(seed)

    # 1 - Run name (the folder name tells which model / config it is)
    phase = "frozen" if unfreeze_last_n_layers == 0 else f"unfreeze{unfreeze_last_n_layers}layers"
    lr_tag = lr_head if unfreeze_last_n_layers == 0 else lr_finetune
    pollution_percent = int(round(pollution_fraction * 100))
    pollution = "clean" if pollution_percent == 0 else f"{pollution_method}{pollution_percent}"
    run_dir = make_run_dir(f"C_mobilenetv2_{phase}_lr{lr_tag:g}_{pollution}")
    print("Results folder :", run_dir)

    # 2 - Pollution (train only, on raw uint8 pixels), THEN normalization
    if pollution_fraction > 0:
        X_train = pollute_train(X_train, pollution_method, pollution_fraction, seed)
    X_train_std = preprocess_input(X_train)
    X_valid_std = preprocess_input(X_valid)
    X_test_std = preprocess_input(X_test)

    # 3 - Model and training
    model, backbone = build_model(num_classes=len(class_names))
    histories = []
    start = time.perf_counter()

    def train_stage(lr, epochs):
        # compile is mandatory after any change of trainable
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        history = model.fit(
            X_train_std, y_train,
            validation_data=(X_valid_std, y_valid),
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stop], verbose=2,
        )
        histories.append(history.history)

    # Phase 1 : frozen backbone, only the head learns
    train_stage(lr_head, epochs_head)

    # Phase 2 : unfreeze the last layers and continue with a much lower lr
    if unfreeze_last_n_layers > 0:
        set_unfrozen_layers(backbone, unfreeze_last_n_layers)
        train_stage(lr_finetune, epochs_finetune)

    training_time = time.perf_counter() - start

    # 4 - Evaluation on the common test set
    y_pred = model.predict(X_test_std, verbose=0).argmax(axis=1)
    labels = np.arange(len(class_names))
    test_output = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classes": list(class_names),
    }

    # 5 - Save everything in results/<run_name>/
    model.save_weights(str(run_dir / "weights.weights.h5"))

    def merge(key):  # concatenate the curves of the two phases
        return [v for h in histories for v in h[key]]

    pd.DataFrame({"epoch": range(1, len(merge("loss")) + 1),
                  "loss": merge("loss"), "accuracy": merge("accuracy")}
                 ).to_csv(run_dir / "train_loss.csv", index=False)
    pd.DataFrame({"epoch": range(1, len(merge("val_loss")) + 1),
                  "loss": merge("val_loss"), "accuracy": merge("val_accuracy")}
                 ).to_csv(run_dir / "valid_loss.csv", index=False)

    with open(run_dir / "test_output.json", "w") as f:
        json.dump(test_output, f, indent=2)

    config = {
        "model": "MobileNetV2",
        "track": "C",
        "unfreeze_last_n_layers": unfreeze_last_n_layers,
        "lr_head": lr_head,
        "lr_finetune": lr_finetune,
        "optimizer": "Adam",
        "activation_output": "softmax",
        "batch_size": batch_size,
        "epochs_head": epochs_head,
        "epochs_finetune": epochs_finetune,
        "pollution_method": pollution_method,
        "pollution_percent": pollution_percent,
        "seed": seed,
        "total_params": model.count_params(),
        "trainable_params": count_params(model.trainable_weights),
        "training_time_s": round(training_time, 1),
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return test_output