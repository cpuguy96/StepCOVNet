from abc import ABC

import keras
from keras import layers, models


class AbstractModel(ABC, object):
    def __init__(
        self,
        model_input: keras.KerasTensor | list[keras.KerasTensor],
        model_output: layers.Layer,
        name: str,
    ):
        self.input = model_input
        self.output = model_output
        self.name = name

    @property
    def model(self) -> models.Model:
        return models.Model(self.input, self.output, name=self.name)
