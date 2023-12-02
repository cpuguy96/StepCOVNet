import json
import os
from abc import abstractmethod, ABC
from datetime import datetime
from typing import List, Union

import tensorflow as tf
import transformers
from tensorflow.keras.initializers import he_uniform, Constant, glorot_uniform
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Conv2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPooling2D,
    TimeDistributed,
    GlobalMaxPool1D,
    Activation,
    BatchNormalization,
    concatenate,
    Dense,
    Dropout,
    Input,
    Layer,
)
from tensorflow.keras.models import load_model, Model
from transformers import GPT2Config, TFGPT2Model

from stepcovnet import config, constants

VGGISH_WEIGHTS_PATH = "stepcovnet/pretrained_models/vggish_audioset_weights.h5"


class AbstractModel(ABC, object):
    def __init__(
        self, model_input: Union[Input, List[Input]], model_output: Layer, name: str
    ):
        self.input = model_input
        self.output = model_output
        self.name = name

    @property
    def model(self) -> Model:
        return Model(self.input, self.output, name=self.name)


class ArrowModel(AbstractModel):
    def __init__(
        self, training_config: config.TrainingConfig, name: str = "StepCOVNetArrowModel"
    ):
        arrow_input = Input(
            shape=training_config.arrow_input_shape, name="arrow_input", dtype=tf.int32
        )
        arrow_mask = Input(
            shape=training_config.arrow_mask_shape, name="arrow_mask", dtype=tf.int32
        )
        model_input = [arrow_input, arrow_mask]

        model_output = self._create_arrow_model(
            arrow_input=arrow_input, arrow_mask=arrow_mask
        )

        super(ArrowModel, self).__init__(
            model_input=model_input, model_output=model_output, name=name
        )

    @abstractmethod
    def _create_arrow_model(self, arrow_input: Input, arrow_mask: Input) -> Layer:
        ...


class AudioModel(AbstractModel):
    def __init__(
        self, training_config: config.TrainingConfig, name: str = "StepCOVNetAudioModel"
    ):
        model_input = Input(
            shape=training_config.audio_input_shape,
            name="audio_input",
            dtype=tf.float64,
        )

        model_output = self._create_audio_model(training_config, model_input)

        super(AudioModel, self).__init__(
            model_input=model_input, model_output=model_output, name=name
        )

    @abstractmethod
    def _create_audio_model(
        self, training_config: config.TrainingConfig, model_input: Input
    ) -> Layer:
        ...


class ClassifierModel(AbstractModel):
    def __init__(
        self,
        training_config: config.TrainingConfig,
        arrow_model: ArrowModel,
        audio_model: AudioModel,
        name="StepCOVNet",
    ):
        model_input = [arrow_model.input, audio_model.input]

        feature_concat = concatenate([arrow_model.output, audio_model.output])
        model = Dense(
            256,
            kernel_initializer=tf.keras.initializers.he_uniform(42),
            bias_initializer=tf.keras.initializers.Zeros(),
        )(feature_concat)
        model = BatchNormalization()(model)
        model = Activation("relu")(model)
        model = Dropout(0.5)(model)

        model_output = Dense(
            constants.NUM_ARROW_COMBS,
            activation="softmax",
            bias_initializer=Constant(value=training_config.init_bias_correction),
            kernel_initializer=glorot_uniform(42),
            dtype=tf.float32,
            name="onehot_encoded_arrows",
        )(model)

        super(ClassifierModel, self).__init__(
            model_input=model_input, model_output=model_output, name=name
        )


class GPT2ArrowModel(ArrowModel):
    def _create_arrow_model(self, arrow_input: Input, arrow_mask: Input) -> Layer:
        gp2_model = PretrainedModels.gpt2_model()
        model_output = gp2_model(arrow_input, attention_mask=arrow_mask)[0]
        # GPT-2 model returns feature maps for avg/max pooling. Using LSTM for additional feature extraction.
        # Might be able to replace this with another method in the future
        return GlobalMaxPool1D()(model_output)


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
    ) -> Model:
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
            aud_input = Input(
                shape=input_shape, name="vggish_input", tensor=input_tensor
            )
        else:
            aud_input = Input(shape=input_shape, name="vggish_input")

        if lookback > 1:
            x = TimeDistributed(
                Conv2D(
                    64,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv1",
                )
            )(aud_input)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")
            )(x)

            # Block 2
            x = TimeDistributed(
                Conv2D(
                    128,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv2",
                )
            )(x)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")
            )(x)

            # Block 3
            x = TimeDistributed(
                Conv2D(
                    256,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv3/conv3_1",
                )
            )(x)
            x = TimeDistributed(
                Conv2D(
                    256,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv3/conv3_2",
                )
            )(x)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")
            )(x)

            # Block 4
            x = TimeDistributed(
                Conv2D(
                    512,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv4/conv4_1",
                )
            )(x)
            x = TimeDistributed(
                Conv2D(
                    512,
                    (3, 3),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                    name="conv4/conv4_2",
                )
            )(x)
            x = TimeDistributed(
                MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool4")
            )(x)
        else:
            # Block 1
            x = Conv2D(
                64,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv1",
            )(aud_input)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(x)

            # Block 2
            x = Conv2D(
                128,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv2",
            )(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(x)

            # Block 3
            x = Conv2D(
                256,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv3/conv3_1",
            )(x)
            x = Conv2D(
                256,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv3/conv3_2",
            )(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(x)

            # Block 4
            x = Conv2D(
                512,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv4/conv4_1",
            )(x)
            x = Conv2D(
                512,
                (3, 3),
                strides=(1, 1),
                activation="relu",
                padding="same",
                name="conv4/conv4_2",
            )(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool4")(x)

        if pooling == "avg":
            if lookback > 1:
                x = TimeDistributed(GlobalAveragePooling2D())(x)
            else:
                x = GlobalAveragePooling2D()(x)
        elif pooling == "max":
            if lookback > 1:
                x = TimeDistributed(GlobalMaxPooling2D())(x)
            else:
                x = GlobalMaxPooling2D()(x)

        # Create model
        model = Model(aud_input, x, name="VGGish")

        # load weights
        if load_weights:
            model.load_weights(VGGISH_WEIGHTS_PATH)

        if freeze:
            for layer in model.layers:
                layer.trainable = False

        return model


class SimpleArrowModel(ArrowModel):
    def _create_arrow_model(self, arrow_input: Input, arrow_mask: Input) -> Layer:
        x = LSTM(64, kernel_initializer="glorot_normal", return_sequences=False)(
            inputs=arrow_input, mask=arrow_mask
        )
        return x


class SimpleAudioModel(AudioModel):
    def _create_audio_model(
        self, training_config: config.TrainingConfig, model_input: Input
    ) -> Layer:
        raise NotImplementedError


class StepCOVNetModel(object):
    def __init__(
        self,
        model_root_path: str,
        model_name: str = "StepCOVNet",
        model: Model = None,
        metadata: dict = None,
    ):
        self.model_root_path = os.path.abspath(model_root_path)
        self.model_name = model_name
        self.model: Model = model
        self.metadata = metadata

    def build_metadata_from_training_config(
        self, training_config: config.TrainingConfig
    ) -> dict:
        self.metadata = {
            "model_name": self.model_name,
            "creation_time": datetime.utcnow().strftime("%b %d %Y %H:%M:%S UTC"),
            "training_config": {
                "limit": training_config.limit,
                "lookback": training_config.lookback,
                "difficulty": training_config.difficulty,
                "tokenizer_name": training_config.tokenizer_name,
                "hyperparameters": str(training_config.hyperparameters),
            },
            "dataset_config": training_config.dataset_config,
        }
        return self.metadata

    @classmethod
    def load(
        cls, input_path: str, retrained: bool = False, compile_model: bool = False
    ):
        metadata = json.load(open(os.path.join(input_path, "metadata.json"), "r"))
        model_name = metadata["model_name"]
        model_path = (
            os.path.join(input_path, model_name + "_retrained")
            if retrained
            else os.path.join(input_path, model_name)
        )
        model = load_model(model_path, compile=compile_model)
        return cls(
            model_root_path=input_path,
            model_name=model_name,
            model=model,
            metadata=metadata,
        )


class VggishAudioModel(AudioModel):
    def _create_audio_model(
        self, training_config: config.TrainingConfig, model_input: Input
    ) -> tf.keras.layers.Layer:
        # Channel reduction
        if training_config.dataset_config["NUM_CHANNELS"] > 1:
            vggish_input = TimeDistributed(
                Conv2D(
                    1,
                    (1, 1),
                    strides=(1, 1),
                    activation="linear",
                    padding="same",
                    kernel_initializer=he_uniform(42),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    image_shape=model_input.shape[1:],
                    data_format="channels_last",
                    name="channel_reduction",
                )
            )(model_input)
        else:
            vggish_input = model_input
        vggish_input = BatchNormalization()(vggish_input)
        vggish_model = PretrainedModels.vggish_model(
            input_shape=training_config.audio_input_shape,
            input_tensor=vggish_input,
            lookback=training_config.lookback,
        )
        model_output = vggish_model(vggish_input)
        # VGGish model returns feature maps for avg/max pooling. Using LSTM for additional feature extraction.
        # Might be able to replace this with another method in the future
        return Bidirectional(
            LSTM(128, return_sequences=False, kernel_initializer=glorot_uniform(42))
        )(model_output)
