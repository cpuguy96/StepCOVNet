from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer

from stepcovnet import config
from stepcovnet.model.AbstractModel import AbstractModel


class AudioModel(AbstractModel):
    def __init__(
        self, training_config: config.TrainingConfig, name: str = "StepCOVNetAudioModel"
    ):
        model_input = Input(
            shape=training_config.audio_input_shape,
            name="audio_input",
            dtype=tf.float64,
        )

        model_output = self._create_audio_model(training_config, model_input)

        super(AudioModel, self).__init__(
            model_input=model_input, model_output=model_output, name=name
        )

    @abstractmethod
    def _create_audio_model(
        self, training_config: config.TrainingConfig, model_input: Input
    ) -> Layer:
        ...
