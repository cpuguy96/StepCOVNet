import keras
from keras import layers

from stepcovnet.model.ArrowModel import ArrowModel


class SimpleArrowModel(ArrowModel):
    def _create_arrow_model(
        self, arrow_input: keras.KerasTensor, arrow_mask: keras.KerasTensor
    ) -> layers.Layer:
        x = layers.LSTM(64, kernel_initializer="glorot_normal", return_sequences=False)(
            inputs=arrow_input, mask=arrow_mask
        )
        return x
