import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def save_results(run_name, model, history, y_test, scores, class_names, config):
    folder = Path("results") / run_name
    folder.mkdir(parents=True, exist_ok=True)

    predictions = np.argmax(scores, axis=1)
    labels = list(range(len(class_names)))
    results = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(
            y_test, predictions, labels=labels, average="macro", zero_division=0
        )),
        "class_names": list(class_names),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
    }

    with (folder / "test_output.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    pd.DataFrame({"loss": history.history["loss"]}).to_csv(
        folder / "train_loss.csv", index=False
    )
    pd.DataFrame({"loss": history.history["val_loss"]}).to_csv(
        folder / "valid_loss.csv", index=False
    )

    with (folder / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    model.save_weights(str(folder / "model.weights.h5"))
    return results
