"""Training reproducibility helpers."""

import os
import random

import keras
import numpy as np
import tensorflow as tf


def apply_training_seed(seed: int) -> None:
    """Set Python, NumPy, TensorFlow, and Keras RNG state before model build.

    Call once at the start of a training run, before constructing models or
    reading data that depends on randomness.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, tf.errors.NotImplementedError):
        pass
