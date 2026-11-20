from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMAGE_SIZE = 128

SPLIT_FOLDERS = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}


def load_images(split_name, with_padding=False):
    splits = pd.read_csv("splits.csv")
    class_names = sorted(splits["class_name"].unique())
    subset = splits[splits["split"] == split_name]

    folder = Path("data") / SPLIT_FOLDERS[split_name]
    target_size = (IMAGE_SIZE, IMAGE_SIZE)
    resampling = Image.Resampling.LANCZOS

    X = np.empty((len(subset), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    y = np.empty(len(subset), dtype=np.int64)

    for i, (_, row) in enumerate(subset.iterrows()):
        image_path = folder / row["relative_path"]

        with Image.open(image_path) as file:
            image = file.convert("RGB")

            if with_padding:
                image = ImageOps.pad(
                    image, target_size, method=resampling, color="black"
                )
            else:
                image = image.resize(target_size, resample=resampling)

            X[i] = np.array(image)

        y[i] = class_names.index(row["class_name"])

    return X, y, class_names