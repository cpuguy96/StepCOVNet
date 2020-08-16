from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from transformers import GPT2Config
from transformers import TFGPT2Model

from stepcovnet.common.tf_config import MIXED_PRECISION_POLICY

VGGISH_WEIGHTS_PATH = "../pretrained_models/vggish_audioset_weights.h5"


class PretrainedModels(object):
    @staticmethod
    def gpt2_model(freeze=True, configuration=None):
        if configuration is None:
            configuration = GPT2Config()
        gp2_model = TFGPT2Model.from_pretrained('gpt2', config=configuration)
        if freeze:
            for layer in gp2_model.layers:
                layer.trainable = False
        for layer in gp2_model.layers:
            layer._set_dtype_policy(MIXED_PRECISION_POLICY)
        return gp2_model

    @staticmethod
    def vggish_model(input_shape, load_weights=True, pooling='avg', freeze=True, input_tensor=None, lookback=1):
        """
        An implementation of the VGGish architecture.
        :param input_shape:
        :param lookback:
        :param freeze:
        :param load_weights: if load weights
        :param input_tensor: input_layer
        :param pooling: pooling type over the non-top network, 'avg' or 'max'
        :return: A Tensorflow Functional API model instance.
        """

        if input_tensor is None:
            aud_input = Input(shape=input_shape, name='vggish_input', tensor=input_tensor)
        else:
            aud_input = Input(shape=input_shape, name='vggish_input')

        if lookback > 1:
            x = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1',
                                       dtype=MIXED_PRECISION_POLICY))(
                aud_input)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1', dtype=MIXED_PRECISION_POLICY))(x)

            # Block 2
            x = TimeDistributed(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2',
                                       dtype=MIXED_PRECISION_POLICY))(x)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2', dtype=MIXED_PRECISION_POLICY))(x)

            # Block 3
            x = TimeDistributed(
                Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1',
                       dtype=MIXED_PRECISION_POLICY))(x)
            x = TimeDistributed(
                Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2',
                       dtype=MIXED_PRECISION_POLICY))(x)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3', dtype=MIXED_PRECISION_POLICY))(x)

            # Block 4
            x = TimeDistributed(
                Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1',
                       dtype=MIXED_PRECISION_POLICY))(x)
            x = TimeDistributed(
                Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2',
                       dtype=MIXED_PRECISION_POLICY))(x)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4', dtype=MIXED_PRECISION_POLICY))(x)
        else:
            # Block 1
            x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1',
                       dtype=MIXED_PRECISION_POLICY)(aud_input)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1', dtype=MIXED_PRECISION_POLICY)(x)

            # Block 2
            x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2',
                       dtype=MIXED_PRECISION_POLICY)(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2', dtype=MIXED_PRECISION_POLICY)(x)

            # Block 3
            x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1',
                       dtype=MIXED_PRECISION_POLICY)(x)
            x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2',
                       dtype=MIXED_PRECISION_POLICY)(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3', dtype=MIXED_PRECISION_POLICY)(x)

            # Block 4
            x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1',
                       dtype=MIXED_PRECISION_POLICY)(x)
            x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2',
                       dtype=MIXED_PRECISION_POLICY)(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4', dtype=MIXED_PRECISION_POLICY)(x)

        if pooling == 'avg':
            if lookback > 1:
                x = TimeDistributed(GlobalAveragePooling2D(dtype=MIXED_PRECISION_POLICY))(x)
            else:
                x = GlobalAveragePooling2D(dtype=MIXED_PRECISION_POLICY)(x)
        elif pooling == 'max':
            if lookback > 1:
                x = TimeDistributed(GlobalMaxPooling2D(dtype=MIXED_PRECISION_POLICY))(x)
            else:
                x = GlobalMaxPooling2D(dtype=MIXED_PRECISION_POLICY)(x)

        # Create model
        model = Model(aud_input, x, name='VGGish')

        # load weights
        if load_weights:
            model.load_weights(VGGISH_WEIGHTS_PATH)

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        return model
