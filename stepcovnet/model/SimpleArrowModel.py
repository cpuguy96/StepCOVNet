from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LSTM

from stepcovnet.model.ArrowModel import ArrowModel


class SimpleArrowModel(ArrowModel):
    def _create_arrow_model(self, arrow_input: Input, arrow_mask: Input) -> Layer:
        x = LSTM(64, kernel_initializer='glorot_normal', return_sequences=False)(inputs=arrow_input,
                                                                                 mask=arrow_mask)
        return x
