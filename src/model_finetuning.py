# approach C : fine-tuning
# Unfreeze the last layers of a backbone,
# possibly a second model

# split script : data_load.py : load_images(split_name, with_padding=False)
'''Training script for model C : fine-tuning,
metrics :

'''
from src.data_load import load_images
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_model(input_shape=(128, 128, 3)):
    """Load MobileNetV2 pre-trained on ImageNet, without its classification head."""
    return MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )


def build_model(num_classes=8):
    backbone = load_model()
    backbone.trainable = False  # phase 1 : the backbone is frozen

    # Model input : 128x128 RGB images
    inputs = layers.Input(shape=(128, 128, 3))

    # Pass the inputs through the backbone.
    # training=False keeps the BatchNorm layers in inference mode.
    x = backbone(inputs, training=False)

    # The backbone returns a 3D feature map, GlobalAveragePooling2D turns it into a vector
    x = layers.GlobalAveragePooling2D()(x)

    # New classification head : one probability per class
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.summary()
    return model


if __name__ == "__main__":
    # 1 - Loading
    X_train, y_train, class_name = load_images("train")
    # X_valid, y_valid, class_name_valid = load_images("valid")
    print("Shape of X_train : ", X_train.shape)  # result : (4829, 128, 128, 3)

    # 2 - Normalization : MobileNetV2 expects pixels in [-1, 1]
    # preprocess_input only transforms X, the labels stay unchanged
    X_train_std = preprocess_input(X_train)
    # X_valid_std = preprocess_input(X_valid)

    # 3 - Model building and summary
    build_model()
    