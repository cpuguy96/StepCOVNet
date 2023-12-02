from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer

from stepcovnet import config
from stepcovnet.model.AbstractModel import AbstractModel


class ArrowModel(AbstractModel):
    def __init__(
        self, training_config: config.TrainingConfig, name: str = "StepCOVNetArrowModel"
    ):
        arrow_input = Input(
            shape=training_config.arrow_input_shape, name="arrow_input", dtype=tf.int32
        )
        arrow_mask = Input(
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
    def _create_arrow_model(self, arrow_input: Input, arrow_mask: Input) -> Layer:
        ...
