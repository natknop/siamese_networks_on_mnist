"""Training 3 conv layers with one layer for fine-tuning on 0-5 classes"""
from typing import Optional

import gin

from src import get_regular_model
import tensorflow as tf

from src.data_loader import load_mnist, select_subdataset, dataset_to_tensors
from src.model import AdditionalModelValidation


@gin.configurable()
def train_regular_model_0_5(batch_size: int, epochs: int, steps_per_epoch: Optional[int], exp_name: str):
    model = get_regular_model(output_shape=10)
    print(model.summary())
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(x_train.shape, x_test.shape)
    x_train_0_5, y_train_0_5 = select_subdataset(x_train, y_train, [0, 1, 2, 3, 4, 5])
    ds_train_0_5 = dataset_to_tensors(x_train_0_5, y_train_0_5, classes_depth=10, batch_size=batch_size, shuffle=True)

    x_test_0_5, y_test_0_5 = select_subdataset(x_test, y_test, [0, 1, 2, 3, 4, 5])
    ds_test_0_5 = dataset_to_tensors(x_test_0_5, y_test_0_5, classes_depth=10, batch_size=1000, shuffle=True)

    x_test_6_9, y_test_6_9 = select_subdataset(x_test, y_test, [6, 7, 8, 9])
    ds_test_6_9 = dataset_to_tensors(x_test_6_9, y_test_6_9, classes_depth=10, batch_size=1000, shuffle=True)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=f"regular_model_0_5/{exp_name}")
    val_callback = AdditionalModelValidation(val_ds=ds_test_6_9, name="test_6_9")

    model.fit(ds_train_0_5, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=ds_test_0_5,
              callbacks=[tb_callback, val_callback])


if __name__ == '__main__':
    gin.parse_config_file("configs/regular_model_training.gin")
    train_regular_model_0_5(batch_size=100, epochs=200, steps_per_epoch=None, exp_name="test")


