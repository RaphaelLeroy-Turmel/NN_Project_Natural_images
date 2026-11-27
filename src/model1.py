from tensorflow import keras
from tensorflow.keras import layers

from src.data_load import IMAGE_SIZE

CONV_FILTERS = (32, 64, 128, 128)
ACTIVATION = "relu"
DENSE_UNITS = 128
DROPOUT_RATE = 0.3
ARCHITECTURE = "4x [Conv2D-BatchNorm-ReLU-MaxPool] + GlobalAveragePooling + Dense + Dropout"


def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_model(n_classes, augment=False, dropout_rate=DROPOUT_RATE):
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    if augment:
        x = build_augmentation()(x)
    for filters in CONV_FILTERS:
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(ACTIVATION)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(DENSE_UNITS, activation=ACTIVATION)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_from_scratch")
