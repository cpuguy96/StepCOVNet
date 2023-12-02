import tensorflow as tf
from tensorflow.keras.initializers import Constant, glorot_uniform
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    concatenate,
    Dense,
    Dropout,
)

from stepcovnet import config
from stepcovnet.common.constants import NUM_ARROW_COMBS
from stepcovnet.model.AbstractModel import AbstractModel
from stepcovnet.model.ArrowModel import ArrowModel
from stepcovnet.model.AudioModel import AudioModel


class ClassifierModel(AbstractModel):
    def __init__(
        self,
        training_config: config.TrainingConfig,
        arrow_model: ArrowModel,
        audio_model: AudioModel,
        name="StepCOVNet",
    ):
        model_input = [arrow_model.input, audio_model.input]

        feature_concat = concatenate([arrow_model.output, audio_model.output])
        model = Dense(
            256,
            kernel_initializer=tf.keras.initializers.he_uniform(42),
            bias_initializer=tf.keras.initializers.Zeros(),
        )(feature_concat)
        model = BatchNormalization()(model)
        model = Activation("relu")(model)
        model = Dropout(0.5)(model)

        model_output = Dense(
            NUM_ARROW_COMBS,
            activation="softmax",
            bias_initializer=Constant(value=training_config.init_bias_correction),
            kernel_initializer=glorot_uniform(42),
            dtype=tf.float32,
            name="onehot_encoded_arrows",
        )(model)

        super(ClassifierModel, self).__init__(
            model_input=model_input, model_output=model_output, name=name
        )
