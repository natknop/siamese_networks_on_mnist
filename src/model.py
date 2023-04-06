import gin
import tensorflow as tf


@gin.configurable()
def get_main_conv_block(output_shape: int):
    input_x = tf.keras.layers.Input((28, 28))

    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same')(input_x)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)

    x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)

    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_shape, activation='relu')(x)
    return tf.keras.models.Model(input_x, x)


@gin.configurable()
def get_regular_model(output_shape: int):
    input_x = tf.keras.layers.Input((28, 28))

    x = get_main_conv_block()(input_x)

    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.models.Model(input_x, x)
