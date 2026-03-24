from __future__ import annotations

import tensorflow as tf


def build_model(input_shape: tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """Build architecture compatible with the provided weights-only model.h5.

    This matches a VGG19 encoder + decoder (U-Net style) head.
    """

    inputs = tf.keras.Input(shape=input_shape, name="input_1")

    encoder = tf.keras.applications.VGG19(
        include_top=False,
        weights=None,
        input_tensor=inputs,
    )

    skip1 = encoder.get_layer("block1_conv2").output
    skip2 = encoder.get_layer("block2_conv2").output
    skip3 = encoder.get_layer("block3_conv4").output
    skip4 = encoder.get_layer("block4_conv4").output
    x = encoder.get_layer("block5_conv4").output

    x = tf.keras.layers.Conv2DTranspose(
        128,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name="conv2d_transpose",
    )(x)
    x = tf.keras.layers.Concatenate(name="concatenate")([x, skip4])
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_1")(x)

    x = tf.keras.layers.Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name="conv2d_transpose_1",
    )(x)
    x = tf.keras.layers.Concatenate(name="concatenate_1")([x, skip3])
    x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_2")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_3")(x)

    x = tf.keras.layers.Conv2DTranspose(
        32,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name="conv2d_transpose_2",
    )(x)
    x = tf.keras.layers.Concatenate(name="concatenate_2")([x, skip2])
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d_4")(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv2d_5")(x)

    x = tf.keras.layers.Conv2DTranspose(
        16,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name="conv2d_transpose_3",
    )(x)
    x = tf.keras.layers.Concatenate(name="concatenate_3")([x, skip1])
    x = tf.keras.layers.Dropout(0.3, name="dropout_3")(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv2d_6")(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv2d_7")(x)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="conv2d_8")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg19_unet")
