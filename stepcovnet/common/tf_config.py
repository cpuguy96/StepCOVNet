from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

MIXED_PRECISION_POLICY = tf.keras.mixed_precision.Policy('mixed_float16')
# Disabling global mixed precision since there is a bug in huggingface when using mixed precision in Tensorflow
# Re-enabling once the issue has been resolved: https://github.com/huggingface/transformers/issues/3320
# Update 1/3/2021: Temporary workaround is to tf.cast attention_mask and head_mask on lines 115 and 122 in
# modeling_tf_gpt2.py within in the transformers package. I will make a pull request to the source package to fix this.
tf.keras.mixed_precision.set_global_policy(MIXED_PRECISION_POLICY)

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)


# work around to get above configurations initialized during runtime
def tf_init():
    pass
