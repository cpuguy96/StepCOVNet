from abc import ABC
from typing import List
from typing import Union

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class AbstractModel(ABC, object):
    def __init__(self, model_input: Union[Input, List[Input]], model_output: Layer, name: str):
        self.input = model_input
        self.output = model_output
        self.name = name

    @property
    def model(self) -> Model:
        return Model(self.input, self.output, name=self.name)
