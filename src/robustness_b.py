import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFilter

from src.data_load import IMAGE_SIZE, load_images
from src.model_b import build_model_b


def degrade_images(images, method, level):
    modified_images = np.empty_like(images)

    for i, pixels in enumerate(images):
        image = Image.fromarray(pixels)
        width, height = image.size

        if method == "flou":
            image = image.filter(ImageFilter.GaussianBlur(radius=level))
        elif method == "pixelisation":
            image = image.resize((level, level), Image.Resampling.BILINEAR)
            image = image.resize((width, height), Image.Resampling.NEAREST)
        elif method == "masque":
            left = (width - level) // 2
            top = (height - level) // 2
            ImageDraw.Draw(image).rectangle(
                (left, top, left + level - 1, top + level - 1), fill="black"
            )
        else:
            raise ValueError(f"Méthode inconnue : {method}")

        modified_images[i] = np.array(image)

    return modified_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize", choices=["direct", "padding"], default="direct")
    args = parser.parse_args()
    source = Path("results") / f"model_b_{args.resize}_v1"
    weights = source / "model.weights.h5"
    if not weights.is_file():
        raise FileNotFoundError(f"Poids introuvables : {weights}")

    folder = Path("results") / f"model_b_{args.resize}_robustness_v1"
    folder.mkdir(parents=True, exist_ok=True)
    X_test, y_test, _ = load_images("test", with_padding=args.resize == "padding")

    model = build_model_b(weights=None)
    model.load_weights(str(weights))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    experiments = [
        ("flou", [0, 1, 2, 3], "Rayon du flou (px)"),
        ("pixelisation", [1, 2, 4, 8], "Largeur d’un bloc (px)"),
        ("masque", [0, 24, 40, 64], "Côté du carré noir (px)"),
    ]

    clean_loss, clean_accuracy = model.evaluate(X_test, y_test, verbose=0)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for col, (method, levels, axis_label) in enumerate(experiments):
        accuracies = []
        losses = []
        for step, level in enumerate(levels):
            applied_level = IMAGE_SIZE // level if method == "pixelisation" else level
            if step == 0:
                loss, accuracy = clean_loss, clean_accuracy
            else:
                altered = degrade_images(X_test, method, applied_level)
                loss, accuracy = model.evaluate(altered, y_test, verbose=0)
                del altered

            rows.append({
                "method": method,
                "level": level,
                "intermediate_size_px": applied_level if method == "pixelisation" else None,
                "accuracy": float(accuracy),
                "loss": float(loss),
                "accuracy_drop_pp": float(100 * (clean_accuracy - accuracy)),
            })
            accuracies.append(100 * accuracy)
            losses.append(loss)
            print(f"{method:13} {level:2}px | accuracy {accuracy:.2%} | loss {loss:.4f}")

        axes[0, col].plot(levels, accuracies, "o-")
        axes[0, col].set_ylim(0, 100)
        axes[0, col].set_title(method.capitalize())
        axes[0, col].set_ylabel("Accuracy (%)")
        axes[1, col].plot(levels, losses, "o-")
        axes[1, col].set_ylabel("Loss")
        axes[1, col].set_xlabel(axis_label)
        for ax in axes[:, col]:
            ax.set_xticks(levels)
            ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(folder / "robustness_curves.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(folder / "robustness_metrics.csv", index=False)
    with (folder / "test_output.json").open("w", encoding="utf-8") as file:
        json.dump({"n_images": len(y_test), "conditions": rows}, file, indent=4)
    with (folder / "config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump({
            "weights": weights.as_posix(),
            "resize": args.resize,
            "image_size": IMAGE_SIZE,
            "split": "test",
            "blur_radius_px": experiments[0][1],
            "pixel_block_width_px": experiments[1][1],
            "black_square_side_px": experiments[2][1],
        }, file)
