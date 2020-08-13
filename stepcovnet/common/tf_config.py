from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# tf.keras.mixed_precision.experimental.set_policy(policy)

# tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)


# work around to get above configurations initialized during runtime
def tf_init():
    pass
