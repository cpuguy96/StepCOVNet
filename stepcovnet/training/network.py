import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from stepcovnet.training.architectures import back
from stepcovnet.training.architectures import front
from stepcovnet.training.architectures import paper_back
from stepcovnet.training.architectures import paper_front
from stepcovnet.training.architectures import pretrained_back
from stepcovnet.training.architectures import pretrained_time_back
from stepcovnet.training.architectures import time_back
from stepcovnet.training.architectures import time_front


def get_init_bias_correction(num_pos, num_all):
    num_neg = num_all - num_pos
    return np.log(num_pos / num_neg)


def get_pretrained_front(model, x_input):
    for i, layer in enumerate(model.layers):
        layer._name = layer.name + str("_pre")
        if isinstance(layer, (
                tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.GlobalMaxPooling2D)):
            # should never start with flatten or pool layer
            return tf.keras.layers.GlobalAveragePooling2D()(
                Model(inputs=x_input, outputs=model.layers[i - 1].output).output)
        else:
            layer.trainable = False
    return tf.keras.layers.GlobalAveragePooling2D()(Model(inputs=x_input, outputs=model.layers[-1].output).output)


def build_stepcovnet(input_shape,
                     timeseries=False,
                     pretrained_model=None,
                     model_type="normal",
                     output_bias_init=tf.keras.initializers.Constant(value=0.0),
                     name="StepCOVNet"):
    if timeseries:
        x_input = Input(input_shape[1:], dtype=tf.float32, name="log_mel_input")
    else:
        x_input = Input(input_shape, dtype=tf.float32, name="log_mel_input")

    inputs = x_input if pretrained_model is None else pretrained_model.layers[0].input

    if model_type == "paper":
        """ The net work architecture given by the paper: Dance Dance Convolution (https://arxiv.org/abs/1703.06891)
        """
        if input_shape[1] == 1:
            channel_order = 'channels_first'
        else:
            channel_order = 'channels_last'

        x = paper_front(x_input,
                        input_shape[1:],
                        channel_order=channel_order)
        x = Flatten()(x)
        x = paper_back(x, input_shape[0])
        x = Dense(1, activation="sigmoid",
                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=(1 / x.get_shape()[-1])),
                  bias_initializer=tf.keras.initializers.Constant(value=0.0),
                  dtype=tf.float32)(x)
    else:
        if timeseries:
            if input_shape[1] == 1:
                channel = 1
                channel_order = 'channels_first'
            else:
                channel = -1
                channel_order = 'channels_last'

            if pretrained_model is not None:
                x = get_pretrained_front(pretrained_model, inputs)
                x = pretrained_time_back(x, input_shape[0])
            else:
                x = time_front(x_input,
                               input_shape[1:],
                               channel_order=channel_order,
                               channel=channel)
                x = Flatten()(x)
                x = time_back(x, input_shape[0])
        else:
            if input_shape[0] == 1:
                channel = 1
                channel_order = 'channels_first'
            else:
                channel = -1
                channel_order = 'channels_last'

            if pretrained_model is not None:
                x = get_pretrained_front(pretrained_model, inputs)
                x = pretrained_back(x)
            else:
                x = front(x_input,
                          input_shape,
                          channel_order=channel_order,
                          channel=channel)
                x = Flatten()(x)
                x = back(x)

        x = Dense(1, activation="sigmoid",
                  kernel_initializer=tf.keras.initializers.glorot_uniform(42),
                  bias_initializer=output_bias_init,
                  dtype=tf.float32)(x)

    return Model(inputs=inputs, outputs=x, name=name)
