"""Track C : run one experiment (train, evaluate on validation, save everything in results/).

Protocol : the test set is NOT used while tuning (evaluate_test=False). It is evaluated only
for the finalists, once, with evaluate_on_test (or evaluate_test=True for the pollution study).
"""
import gc
import json
import shutil
import time

import numpy as np
import pandas as pd
import yaml  # pip install pyyaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tensorflow import keras

from src.degradation import pollute_train
from src.model_finetuning import build_model, set_unfrozen_layers
from src.runs import RESULTS_DIR, run_name

# Parameters saved in config.yaml and compared when a finished run is requested again
HYPERPARAMS = (
    "unfreeze_last_n_layers", "lr_head", "lr_finetune", "epochs_head", "epochs_finetune",
    "batch_size", "pollution_method", "pollution_fraction", "seed", "tag", "pretrained",
)


def _evaluate(model, X, y, class_names, batch_size=32):
    """Loss, accuracy, macro F1 and confusion matrix on one split. Returns (metrics, y_pred)."""
    y = np.asarray(y)
    probs = model.predict(X, batch_size=batch_size, verbose=0) # distribution of probability for each class
    y_pred = probs.argmax(axis=1) #  we take the most probable class
    labels = np.arange(len(class_names)) 
    p_true = np.clip(probs[np.arange(len(y)), y], 1e-7, 1.0)  # we take the probability for the actual true class y , 
    # and we limit the value between 1e-7 et 1.0 because if p = 0 then  log(p)= -inf (for the cross-entropy later)
    metrics = {
        "loss": float(-np.log(p_true).mean()),  # cross-entropy
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "classes": list(class_names),
    }
    return metrics, y_pred


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _check_same_config(final_dir, config):
    """A finished folder with the same name must come from the same hyperparameters."""
    saved = yaml.safe_load((final_dir / "config.yaml").read_text())
    differences = {k: (saved.get(k), config[k]) for k in HYPERPARAMS if saved.get(k) != config[k]}
    if differences:
        raise ValueError(
            f"{final_dir.name} already exists with another config (saved, requested) : {differences}. "
            "Use the `tag` parameter to give this variant its own folder."
        )


def evaluate_on_test(name, X_test, y_test, class_names, batch_size=32):
    """Evaluate a finished run on the test set.

    The test is read once : if test_output.json already exists, it is returned as it is.
    """
    final_dir = RESULTS_DIR / name
    out = final_dir / "test_output.json"
    if out.exists():
        return json.loads(out.read_text())
    model, _ = build_model(len(class_names), pretrained=False)  # the weights come from the file
    model.load_weights(str(final_dir / "weights.weights.h5"))
    metrics, _ = _evaluate(model, X_test, y_test, class_names, batch_size)
    _write_json(out, metrics)
    keras.backend.clear_session()
    gc.collect()
    return metrics


def run_experiment(
    X_train, y_train, X_valid, y_valid, class_names,
    X_test=None, y_test=None,
    unfreeze_last_n_layers=0,   # 0 = backbone frozen only (like track B)
    lr_head=1e-3,               # phase 1 : train the new head
    lr_finetune=1e-5,           # phase 2 : very low lr on the unfrozen layers
    epochs_head=10,
    epochs_finetune=10,
    batch_size=32,
    pollution_method=None,      # "flou", "pixelisation", "masque" or None
    pollution_fraction=0.0,     # fraction of the TRAIN images that are degraded
    seed=42,                    # training seed AND pollution seed
    tag="",                     # extra label in the folder name for variants
    evaluate_test=False,        # False while tuning : the test set is not read
    pretrained=True,
):
    """Train one model and save everything in results/<run name>/. Returns the run name.

    The X arrays must contain raw uint8 pixels : the normalization is inside the model.
    A run that is already finished is skipped, so a launcher can be restarted after a crash.
    """
    config = {
        "unfreeze_last_n_layers": unfreeze_last_n_layers,
        "lr_head": lr_head,
        "lr_finetune": lr_finetune,
        "epochs_head": epochs_head,
        "epochs_finetune": epochs_finetune,
        "batch_size": batch_size,
        "pollution_method": pollution_method,
        "pollution_fraction": pollution_fraction,
        "seed": seed,
        "tag": tag,
        "pretrained": pretrained,
    }
    # Check the arguments first : a wrong call must fail now, not after minutes of training
    assert pollution_fraction == 0 or pollution_method is not None, \
        "pollution_method is required when pollution_fraction > 0"
    assert not evaluate_test or (X_test is not None and y_test is not None), \
        "X_test and y_test are required when evaluate_test=True"

    name = run_name(**config)
    final_dir = RESULTS_DIR / name

    # 1 - A finished run is never redone
    if final_dir.is_dir():
        _check_same_config(final_dir, config)
        print(f"[skip] {name} is already done")
        if evaluate_test:
            evaluate_on_test(name, X_test, y_test, class_names, batch_size)
        return name

    # 2 - Everything is written in <name>.partial, renamed only when the run is complete.
    #     A crash therefore never leaves a folder that looks finished.
    partial_dir = RESULTS_DIR / f"{name}.partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)  # leftover of a crashed run
    partial_dir.mkdir(parents=True)
    print("Results folder :", final_dir)

    keras.utils.set_random_seed(seed)

    # 3 - Pollution (train only, on raw uint8 pixels). The pollution seed is the run seed.
    n_polluted = int(round(pollution_fraction * len(X_train)))
    if pollution_fraction > 0:
        X_train = pollute_train(X_train, pollution_method, pollution_fraction, seed)

    # 4 - Model and training
    model, backbone = build_model(num_classes=len(class_names), pretrained=pretrained)
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
            X_train, y_train,
            validation_data=(X_valid, y_valid),
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

    # 5 - Validation outputs (used to choose finalists), test only if asked
    valid_metrics, valid_pred = _evaluate(model, X_valid, y_valid, class_names, batch_size)
    _write_json(partial_dir / "valid_output.json", valid_metrics)
    pd.DataFrame({"y_true": np.asarray(y_valid), "y_pred": valid_pred}).to_csv(
        partial_dir / "valid_predictions.csv", index_label="index"
    )  # lets us compare two variants image by image

    if evaluate_test:
        test_metrics, _ = _evaluate(model, X_test, y_test, class_names, batch_size)
        _write_json(partial_dir / "test_output.json", test_metrics)

    # 6 - Save the rest. config.yaml is written last.
    model.save_weights(str(partial_dir / "weights.weights.h5"))

    def merge(key):  # concatenate the curves of the two phases
        return [v for h in histories for v in h[key]]

    n_epochs = len(merge("loss"))
    pd.DataFrame({"epoch": range(1, n_epochs + 1),
                  "loss": merge("loss"), "accuracy": merge("accuracy")}
                 ).to_csv(partial_dir / "train_loss.csv", index=False)
    pd.DataFrame({"epoch": range(1, n_epochs + 1),
                  "loss": merge("val_loss"), "accuracy": merge("val_accuracy")}
                 ).to_csv(partial_dir / "valid_loss.csv", index=False)

    full_config = {
        "model": "MobileNetV2",
        "track": "C",
        **config,
        "optimizer": "Adam",
        "activation_output": "softmax",
        "polluted_images": n_polluted,
        "total_params": model.count_params(),
        "trainable_params": int(sum(np.prod(tuple(w.shape)) for w in model.trainable_weights)),
        "epochs_run": n_epochs,
        "training_time_s": round(training_time, 1),
    }
    with open(partial_dir / "config.yaml", "w") as f:
        yaml.safe_dump(full_config, f, sort_keys=False)

    # 7 - The run is complete : the folder takes its final name
    partial_dir.rename(final_dir)

    keras.backend.clear_session()
    gc.collect()
    return name
