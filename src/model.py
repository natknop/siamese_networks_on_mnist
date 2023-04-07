import gin
import tensorflow as tf


class AdditionalModelValidation(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, name: str):
        super().__init__()
        self.val_ds = val_ds
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        val_res = self.model.evaluate(self.val_ds)
        print(val_res)
        logs[f"{self.name}_accuracy"] = val_res[0]


@gin.configurable()
def get_main_conv_block(output_shape: int):
    input_x = tf.keras.layers.Input((28, 28, 1))

    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same')(input_x)
    x = tf.keras.layers.MaxPool2D(pool_size=3)(x)

    x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3)(x)

    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_shape, activation='relu')(x)
    return tf.keras.models.Model(input_x, x)


@gin.configurable()
def get_regular_model(output_shape: int):
    input_x = tf.keras.layers.Input((28, 28, 1, ))

    x = get_main_conv_block()(input_x)

    x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    return tf.keras.models.Model(input_x, x)
