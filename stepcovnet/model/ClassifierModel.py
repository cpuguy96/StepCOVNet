import tensorflow as tf
from keras import layers, initializers

from stepcovnet.common.constants import NUM_ARROW_COMBS
from stepcovnet.config.TrainingConfig import TrainingConfig
from stepcovnet.model.AbstractModel import AbstractModel
from stepcovnet.model.ArrowModel import ArrowModel
from stepcovnet.model.AudioModel import AudioModel


class ClassifierModel(AbstractModel):
    def __init__(
        self,
        training_config: TrainingConfig,
        arrow_model: ArrowModel,
        audio_model: AudioModel,
        name="StepCOVNet",
    ):
        model_input = [arrow_model.input, audio_model.input]

        feature_concat = layers.concatenate([arrow_model.output, audio_model.output])
        model = layers.Dense(
            256,
            kernel_initializer=initializers.he_uniform(42),
            bias_initializer=initializers.Zeros(),
        )(feature_concat)
        model = layers.BatchNormalization()(model)
        model = layers.Activation("relu")(model)
        model = layers.Dropout(0.5)(model)

        model_output = layers.Dense(
            NUM_ARROW_COMBS,
            activation="softmax",
            bias_initializer=initializers.Constant(
                value=training_config.init_bias_correction
            ),
            kernel_initializer=initializers.glorot_uniform(42),
            dtype=tf.float32,
            name="onehot_encoded_arrows",
        )(model)

        super(ClassifierModel, self).__init__(
            model_input=model_input, model_output=model_output, name=name
        )
