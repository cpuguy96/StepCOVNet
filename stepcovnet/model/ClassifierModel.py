import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from stepcovnet.common.constants import NUM_ARROW_COMBS
from stepcovnet.model.AbstractModel import AbstractModel


class ClassifierModel(AbstractModel):
    def __init__(self, training_config, arrow_model, audio_model, architecture=None, name="StepCOVNet"):
        model_input = [arrow_model.input, audio_model.input]

        feature_concat = concatenate([arrow_model.output, audio_model.output])
        if architecture is None:
            model = Dense(512,
                          kernel_initializer=tf.keras.initializers.he_uniform(42),
                          bias_initializer=tf.keras.initializers.Constant(value=0.1),
                          )(feature_concat)
            model = BatchNormalization()(model)
            model = Activation('relu')(model)
            model = Dropout(0.5)(model)
        else:
            # TODO: Add support for existing classifier models
            raise NotImplementedError("No support yet for existing architectures")

        model_output = Dense(NUM_ARROW_COMBS, activation="softmax",
                             bias_initializer=Constant(value=training_config.init_bias_correction),
                             kernel_initializer=glorot_uniform(42), dtype=tf.float32, name="onehot_encoded_arrows"
                             )(model)

        super(ClassifierModel, self).__init__(model_input=model_input, model_output=model_output, name=name)
