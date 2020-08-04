from abc import ABC

from tensorflow.keras.models import Model


class AbstractModel(ABC):
    def __init__(self, model_input, model_output, name):
        self.input = model_input
        self.output = model_output
        self.name = name

    @property
    def model(self):
        return Model(self.input, self.output, name=self.name)
