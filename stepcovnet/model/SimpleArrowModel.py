from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer

from stepcovnet.model.ArrowModel import ArrowModel


class SimpleArrowModel(ArrowModel):
    def _create_arrow_model(self, arrow_input: Input, arrow_mask: Input) -> Layer:
        raise NotImplementedError
