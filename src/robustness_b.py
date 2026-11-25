import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.data_load import load_images
from src.model_b import build_model_b


def degrade_images(images, method):
    modified_images = np.empty_like(images)

    for i, pixels in enumerate(images):
        image = Image.fromarray(pixels)
        width, height = image.size

        if method == "flou":
            image = image.filter(ImageFilter.GaussianBlur(radius=1)) # gaussianblur = flou

        elif method == "pixelisation":
            image = image.resize((64, 64), Image.Resampling.BILINEAR) # on réduit l'image en 64x64
            image = image.resize((width, height), Image.Resampling.NEAREST) # puis on réagrandit en 128x128

        elif method == "masque": # carré noir qui cache 10% de l'image
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

if __name__ == "__main__":
    X_test, y_test, class_names = load_images("test", with_padding=False)

    model = build_model_b()
    model.load_weights("results/model_b_direct_v1/model.weights.h5")
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    clean_loss, clean_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Original     | accuracy {clean_accuracy:.2%} | loss {clean_loss:.4f}")

    # Choisir le premier chien du test pour illustrer les transformations.
    dog_id = class_names.index("dog")
    example_index = np.where(y_test == dog_id)[0][0]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    axes[0].imshow(X_test[example_index])
    axes[0].set_title("Original")
    axes[0].axis("off")

    methods = ["flou", "pixelisation", "masque"]

    for col, method in enumerate(methods, start=1):
        X_modified = degrade_images(X_test, method)
        loss, accuracy = model.evaluate(X_modified, y_test, verbose=0)
        drop = 100 * (clean_accuracy - accuracy)

        print(
            f"{method:12} | accuracy {accuracy:.2%} | loss {loss:.4f}"
            f" | baisse de {drop:.2f} points"
        )

        axes[col].imshow(X_modified[example_index])
        axes[col].set_title(method.capitalize())
        axes[col].axis("off")

    plt.tight_layout()
    plt.savefig("model_b_robustness_examples.png", dpi=150)
    plt.show()