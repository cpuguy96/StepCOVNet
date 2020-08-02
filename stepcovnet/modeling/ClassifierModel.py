import tensorflow as tf


class ClassifierModel(object):
    def __init__(self, training_config, arrow_model, audio_model):
        # TODO: implement
        output_bias_init = tf.keras.initializers.Constant(value=training_config.init_bias_correction)
