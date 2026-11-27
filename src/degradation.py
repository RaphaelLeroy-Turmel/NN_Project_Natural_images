"""Image degradation used to pollute the training data.

Shared module : every track must import the same functions, so that
the same pollution is applied in every experiment.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def degrade_images(images, method):
    """Degrade every image of a (N, H, W, 3) uint8 array with the given method."""
    modified_images = np.empty_like(images)

    for i, pixels in enumerate(images):
        image = Image.fromarray(pixels)
        width, height = image.size

        if method == "flou":
            image = image.filter(ImageFilter.GaussianBlur(radius=1))  # gaussian blur

        elif method == "pixelisation":
            image = image.resize((64, 64), Image.Resampling.BILINEAR)  # shrink the image to 64x64
            image = image.resize((width, height), Image.Resampling.NEAREST)  # then enlarge it back

        elif method == "masque":  # black square hiding about 10% of the image
            side = 40
            left = (width - side) // 2
            top = (height - side) // 2
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (left, top, left + side - 1, top + side - 1),
                fill="black",
            )

        else:
            raise ValueError(f"Méthode inconnue : {method}")

        modified_images[i] = np.array(image)

    return modified_images


def pollute_train(X, method, fraction, seed):
    """Degrade a random fraction of the images (to use on the TRAIN set only).

    X must contain raw uint8 pixels (0-255) : pollute BEFORE any normalization.

    The images are chosen from ONE random permutation of the dataset (fixed by the seed),
    so the polluted subsets are nested : with the same seed, the images degraded at 10 %
    are also degraded at 20 % and at 30 %. The curves of a pollution study are then
    not disturbed by a different draw at each level.
    """
    assert X.dtype == np.uint8, f"degrade_images needs uint8 images, got {X.dtype}"
    X_out = X.copy()
    n_polluted = int(round(fraction * len(X)))
    if n_polluted == 0:
        return X_out
    rng = np.random.default_rng(seed)  # fixed seed -> same polluted images at each run
    idx = rng.permutation(len(X))[:n_polluted]
    X_out[idx] = degrade_images(X[idx], method)
    return X_out
