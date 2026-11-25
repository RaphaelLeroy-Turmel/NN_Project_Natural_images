import argparse
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers
from src.data_load import IMAGE_SIZE, load_images # récupère la taille normée (128)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

def build_model_b(weights="imagenet"):
    backbone = keras.applications.MobileNetV2( # Création du MobileNetV2
        input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3),
        weights=weights,
        include_top = False # Retire tete de classification imagenet pour pouvoir ajouter la notre
    )

    backbone.trainable = False # Gel des params de MobileNetV2

    model = keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.Rescaling(1/127.5, offset=-1), # standardisation [0,255] -> [-1,1]
        backbone, # params gelés mobilenetV2
        layers.GlobalAveragePooling2D(),
        layers.Dense(8, activation = "softmax") # softmax transforme les 8 scores en valeurs de proba dont la somme fait 1
    ])

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize", choices=["direct", "padding"], default="direct")
    args = parser.parse_args()
    resize_name = args.resize
    with_padding = resize_name == "padding"
    run_name = f"model_b_{resize_name}_v1"
    run_folder = Path("results") / run_name

    keras.utils.set_random_seed(42)

    X_train, y_train, class_names = load_images("train", with_padding=with_padding)
    X_val, y_val, _ = load_images("validation", with_padding=with_padding) 

    model = build_model_b()
    model.summary()

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 0.001),
        loss = "sparse_categorical_crossentropy", 
        metrics = ["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_val, y_val),
        epochs = 3,
        batch_size = 32
    )

    from src.evaluation import save_results

    X_test, y_test, _ = load_images("test", with_padding=with_padding)
    test_scores = model.predict(X_test, batch_size = 32, verbose = 0)

    config = {
        "model": "MobileNetV2 gelé",
        "resize": resize_name,
        "image_size": IMAGE_SIZE,
        "backbone_weights": "imagenet",
        "backbone_trainable": False,
        "classifier_activation": "softmax",
        "total_params": model.count_params(),
        "trainable_params": sum(int(np.prod(weight.shape)) for weight in model.trainable_weights),
        "seed": 42,
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 0.001
    }

    results = save_results(
        run_name,
        model, history, y_test, test_scores, class_names, config
    )

    print(f"Accuracy test : {results['accuracy']:.2%}")
    print(f"Macro-F1 test : {results['f1_macro']:.2%}")

    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], "o-", label="Train")
    plt.plot(epochs, history.history["val_loss"], "o-", label="Validation")
    plt.title("Loss par époque")
    plt.xlabel("Époque")
    plt.xticks(epochs)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], "o-", label="Train")
    plt.plot(epochs, history.history["val_accuracy"], "o-", label="Validation")
    plt.title("Accuracy par époque")
    plt.xlabel("Époque")
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout() # espace les graphiques
    plt.savefig(run_folder / "learning_curves.png", dpi=150)
    plt.close()

    probabilities = model.predict(X_val, batch_size = 32, verbose = 0)
    predictions = np.argmax(probabilities, axis = 1)
    dog_id = class_names.index("dog")
    cat_id = class_names.index("cat")
    dog_images = y_val == dog_id

    print("Macro-F1 :", f1_score(y_val, predictions, average="macro"))
    print("Rappel dog :", np.mean(predictions[dog_images] == dog_id))
    print("Chiens prédits chat :", np.sum(predictions[dog_images] == cat_id))
    wrong_indices = np.where(predictions != y_val)[0]

    print("Nombre d'erreurs : ", len(wrong_indices))

    for i in wrong_indices[:6]:
        true_class = class_names[y_val[i]]
        predicted_class = class_names[predictions[i]]

        plt.figure(figsize=(3, 3))
        plt.imshow(X_val[i])
        plt.title(f"Vrai : {true_class} | Prédit : {predicted_class}")
        plt.axis("off")
        plt.savefig(run_folder / f"validation_error_{i}.png", dpi=150)
        plt.close()