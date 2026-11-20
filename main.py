from pathlib import Path

import numpy as np
import pandas as pd

from src.data_load import IMAGE_SIZE, SPLIT_FOLDERS, load_images


if __name__ == "__main__":
    splits = pd.read_csv("splits.csv")
    assert splits["relative_path"].is_unique
    assert set(splits["split"]) == set(SPLIT_FOLDERS)

    for split_name, folder_name in SPLIT_FOLDERS.items():
        subset = splits[splits["split"] == split_name]
        for relative_path in subset["relative_path"]:
            path = Path("data") / folder_name / relative_path
            if not path.is_file():
                raise FileNotFoundError(path)
        print(split_name, len(subset))

    X_direct, y_direct, classes = load_images("validation")
    X_padding, y_padding, padding_classes = load_images("validation", with_padding=True)

    assert X_direct.shape == X_padding.shape == (len(y_direct), IMAGE_SIZE, IMAGE_SIZE, 3)
    assert np.array_equal(y_direct, y_padding)
    assert classes == padding_classes
    print("Images de validation :", X_direct.shape)
    print("Classes :", classes)
