from abc import abstractmethod

import keras
import tensorflow as tf
from keras import layers

from stepcovnet.config.TrainingConfig import TrainingConfig
from stepcovnet.model.AbstractModel import AbstractModel


class ArrowModel(AbstractModel):
    def __init__(
        self, training_config: TrainingConfig, name: str = "StepCOVNetArrowModel"
    ):
        arrow_input = layers.Input(
            shape=training_config.arrow_input_shape, name="arrow_input", dtype=tf.int32
        )
        arrow_mask = layers.Input(
            shape=training_config.arrow_mask_shape, name="arrow_mask", dtype=tf.int32
        )
        model_input = [arrow_input, arrow_mask]

        model_output = self._create_arrow_model(
            arrow_input=arrow_input, arrow_mask=arrow_mask
        )

        super(ArrowModel, self).__init__(
            model_input=model_input, model_output=model_output, name=name
        )

    @abstractmethod
    def _create_arrow_model(
        self, arrow_input: keras.KerasTensor, arrow_mask: keras.KerasTensor
    ) -> layers.Layer:
        ...
