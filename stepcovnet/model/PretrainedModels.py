import transformers
from keras import models, layers
from transformers import GPT2Config, TFGPT2Model

VGGISH_WEIGHTS_PATH = "stepcovnet/pretrained_models/vggish_audioset_weights.h5"


class PretrainedModels(object):
    @staticmethod
    def gpt2_model(freeze=True, configuration=None):
        if configuration is None:
            configuration = GPT2Config()
        gp2_model = TFGPT2Model.from_pretrained("gpt2", config=configuration)
        if freeze:
            for top_layer in gp2_model.layers[:]:
                if isinstance(
                    top_layer, transformers.models.gpt2.modeling_tf_gpt2.TFGPT2MainLayer
                ):
                    for block in top_layer.h[:]:
                        block.trainable = False
        return gp2_model

    @staticmethod
    def vggish_model(
        input_shape,
        load_weights=True,
        pooling="avg",
        freeze=True,
        input_tensor=None,
        lookback=1,
    ) -> models.Model:
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
            aud_input = layers.Input(
                shape=input_shape, name="vggish_input", tensor=input_tensor
            )
        else:
            aud_input = layers.Input(shape=input_shape, name="vggish_input")

        if lookback > 1:
            x = layers.TimeDistributed(
                layers.Conv2D(
                    64,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv1",
                )
            )(aud_input)
            x = layers.TimeDistributed(
                layers.MaxPooling2D(
                    (2, 2), strides=(2, 2), padding="same", name="pool1"
                )
            )(x)

            # Block 2
            x = layers.TimeDistributed(
                layers.Conv2D(
                    128,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv2",
                )
            )(x)
            x = layers.TimeDistributed(
                layers.MaxPooling2D(
                    (2, 2), strides=(2, 2), padding="same", name="pool2"
                )
            )(x)

            # Block 3
            x = layers.TimeDistributed(
                layers.Conv2D(
                    256,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv3/conv3_1",
                )
            )(x)
            x = layers.TimeDistributed(
                layers.Conv2D(
                    256,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv3/conv3_2",
                )
            )(x)
            x = layers.TimeDistributed(
                layers.MaxPooling2D(
                    (2, 2), strides=(2, 2), padding="same", name="pool3"
                )
            )(x)

            # Block 4
            x = layers.TimeDistributed(
                layers.Conv2D(
                    512,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv4/conv4_1",
                )
            )(x)
            x = layers.TimeDistributed(
                layers.Conv2D(
                    512,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv4/conv4_2",
                )
            )(x)
            x = layers.TimeDistributed(
                layers.MaxPooling2D(
                    (2, 2), strides=(2, 2), padding="same", name="pool4"
                )
            )(x)
        else:
            # Block 1
            x = layers.Conv2D(
                64,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv1",
            )(aud_input)
            x = layers.MaxPooling2D(
                (2, 2), strides=(2, 2), padding="same", name="pool1"
            )(x)

            # Block 2
            x = layers.Conv2D(
                128,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv2",
            )(x)
            x = layers.MaxPooling2D(
                (2, 2), strides=(2, 2), padding="same", name="pool2"
            )(x)

            # Block 3
            x = layers.Conv2D(
                256,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv3/conv3_1",
            )(x)
            x = layers.Conv2D(
                256,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv3/conv3_2",
            )(x)
            x = layers.MaxPooling2D(
                (2, 2), strides=(2, 2), padding="same", name="pool3"
            )(x)

            # Block 4
            x = layers.Conv2D(
                512,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv4/conv4_1",
            )(x)
            x = layers.Conv2D(
                512,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv4/conv4_2",
            )(x)
            x = layers.MaxPooling2D(
                (2, 2), strides=(2, 2), padding="same", name="pool4"
            )(x)

        if pooling == "avg":
            if lookback > 1:
                x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
            else:
                x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            if lookback > 1:
                x = layers.TimeDistributed(layers.GlobalMaxPooling2D())(x)
            else:
                x = layers.GlobalMaxPooling2D()(x)

        # Create model
        model = models.Model(aud_input, x, name="VGGish")

        # load weights
        if load_weights:
            model.load_weights(VGGISH_WEIGHTS_PATH)

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        return model
