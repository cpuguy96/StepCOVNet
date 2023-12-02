from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from keras import mixed_precision

# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

MIXED_PRECISION_POLICY = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(MIXED_PRECISION_POLICY)

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)


# work around to get above configurations initialized during runtime
def tf_init():
    pass
