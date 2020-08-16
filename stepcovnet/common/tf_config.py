from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

MIXED_PRECISION_POLICY = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# Disabling global mixed precision since there is a bug in huggingface when using mixed precision in Tensorflow
# Re-enabling once the issue has been resolved: https://github.com/huggingface/transformers/issues/3320
# tf.keras.mixed_precision.experimental.set_policy(MIXED_PRECISION_POLICY)

# tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)


# work around to get above configurations initialized during runtime
def tf_init():
    pass
