import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed

from stepcovnet.modeling.AbstractModel import AbstractModel
from stepcovnet.modeling.PretrainedModels import PretrainedModels


class AudioModel(AbstractModel):
    def __init__(self, training_config, architecture=None, name="StepCOVNetAudioModel"):
        model_input = Input(shape=training_config.audio_input_shape, name="audio_input", dtype=tf.float64)

        if architecture is None:
            # Channel reduction
            if training_config.dataset_config["NUM_CHANNELS"] > 1:
                vggish_input = TimeDistributed(Conv2D(1, (1, 1), strides=(1, 1), activation='linear',
                                                      padding='same', kernel_initializer=he_uniform(42),
                                                      bias_initializer=he_uniform(42),
                                                      image_shape=model_input.shape[1:], data_format='channels_last',
                                                      name='channel_reduction')
                                               )(model_input)
            else:
                vggish_input = model_input
            vggish_input = BatchNormalization()(vggish_input)
            vggish_model = PretrainedModels.vggish_model(input_shape=training_config.audio_input_shape,
                                                         input_tensor=vggish_input, lookback=training_config.lookback)
            model_output = vggish_model(vggish_input)
            # VGGish model returns feature maps for avg/max pooling. Using LSTM for additional feature extraction.
            # Might be able to replace this with another method in the future
            model_output = Bidirectional(LSTM(64, return_sequences=True, kernel_initializer=glorot_uniform(42))
                                         )(model_output)
            model_output = GlobalMaxPool1D()(model_output)
        else:
            # TODO: Add support for existing audio models
            raise NotImplementedError("No support yet for existing architectures")

        super(AudioModel, self).__init__(model_input=model_input, model_output=model_output, name=name)
