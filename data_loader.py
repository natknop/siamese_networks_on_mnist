from typing import List, Tuple

import numpy as np
import tensorflow as tf


def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Returns (x_train, y_train), (x_test, y_test) for MNIST dataset"""
    return tf.keras.datasets.mnist.load_data()


def select_subdataset(x: np.ndarray, y: np.ndarray, classes_to_extract: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Select specific classes from input numpy arrays"""
    unique_classes = list(set(classes_to_extract))
    selected_x = []
    selected_y = []
    for cl in unique_classes:
        selected_x.append(x[y == cl])
        selected_y.append(y[y == cl])

    return np.concatenate(selected_x), np.concatenate(selected_y)


def dataset_to_tensors(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    """Convert numpy array to tf.data.Dataset"""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y) * 2, seed=10)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_test_0_3, y_test_0_3 = select_subdataset(x_test, y_test, [0, 1, 2, 3])
    ds_test_0_3 = dataset_to_tensors(x_test_0_3, y_test_0_3, batch_size=1000, shuffle=True)
    print(ds_test_0_3)
