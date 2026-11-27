"""Track C : MobileNetV2 model definition (backbone, classification head, freezing)."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

from src.data_load import IMAGE_SIZE


def load_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pretrained=True):
    """Load MobileNetV2 without its classification head (ImageNet weights if pretrained)."""
    return MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet" if pretrained else None,
    )


def build_model(num_classes=8, pretrained=True):
    """Build normalization + backbone + new classification head. Returns (model, backbone).

    The model takes RAW uint8 images (0-255) : the normalization is the first layer of the
    model. The arrays stay small in memory (uint8) and are converted batch by batch.
    pretrained=False is used to rebuild a model whose weights are loaded from a file.
    """
    backbone = load_model(pretrained=pretrained)
    backbone.trainable = False  # phase 1 : the backbone is frozen

    inputs = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype="uint8")
    # Same operation as mobilenet_v2.preprocess_input : pixels from [0, 255] to [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    # training=False keeps the BatchNorm layers in inference mode
    x = backbone(x, training=False)
    # The backbone returns a 3D feature map, pooling turns it into a vector
    x = layers.GlobalAveragePooling2D()(x)
    # New classification head : one probability per class
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, backbone  # the backbone is returned to be able to unfreeze it later


def set_unfrozen_layers(backbone, n_layers):
    """Unfreeze the last n_layers layers of the backbone (0 = fully frozen).

    n_layers counts layers, not blocks : MobileNetV2 has about 150 layers
    and one block is about 8 to 10 layers.
    """
    if n_layers == 0:
        backbone.trainable = False
        return
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
