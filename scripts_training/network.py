from scripts_training.architectures import *

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model


def build_stepcovnet(input_shape,
                     timeseries=False,
                     extra_input_shape=None,
                     pretrained_model=None,
                     name="StepCOVNet"):
    if timeseries:
        x_input = Input(input_shape[1:], dtype='float16', name="log_mel_input")
    else:
        x_input = Input(input_shape, dtype='float16', name="log_mel_input")

    if extra_input_shape is not None:
        extra_input = Input((extra_input_shape[1],), dtype="float16", name="extra_input")
        inputs = [x_input, extra_input]
    else:
        extra_input = None
        inputs = x_input

    if timeseries:
        if input_shape[1] == 1:
            channel = 1
            channel_order = 'channels_first'
        else:
            channel = -1
            channel_order = 'channels_last'

        if pretrained_model:
            # TODO: Add support for pretrained model
            pass
        x = time_front(x_input,
                       input_shape[1:],
                       channel_order=channel_order,
                       channel=channel)
        x = Flatten()(x)
        x = time_back(x, input_shape[0], extra_input)
    else:
        if input_shape[0] == 1:
            channel = 1
            channel_order = 'channels_first'
        else:
            channel = -1
            channel_order = 'channels_last'

        if pretrained_model:
            # TODO: Add support for pretrained model
            pass
        x = front(x_input,
                  input_shape,
                  channel_order=channel_order,
                  channel=channel)
        x = Flatten()(x)
        x = back(x, extra_input)
    x = Dense(1, activation="sigmoid")(x)

    return Model(inputs=inputs, outputs=x, name=name)
