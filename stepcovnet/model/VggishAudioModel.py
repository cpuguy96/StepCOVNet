import keras
from keras import layers, initializers

from stepcovnet.config.TrainingConfig import TrainingConfig
from stepcovnet.model.AudioModel import AudioModel
from stepcovnet.model.PretrainedModels import PretrainedModels


class VggishAudioModel(AudioModel):
    def _create_audio_model(
        self, training_config: TrainingConfig, model_input: keras.KerasTensor
    ) -> layers.Layer:
        # Channel reduction
        if training_config.dataset_config["NUM_CHANNELS"] > 1:
            vggish_input = layers.TimeDistributed(
                layers.Conv2D(
                    1,
                    (1, 1),
                    strides=(1, 1),
                    activation="linear",
                    padding="same",
                    kernel_initializer=initializers.he_uniform(42),
                    bias_initializer=initializers.Zeros(),
                    image_shape=model_input.shape[1:],
                    data_format="channels_last",
                    name="channel_reduction",
                )
            )(model_input)
        else:
            vggish_input = model_input
        vggish_input = layers.BatchNormalization()(vggish_input)
        vggish_model = PretrainedModels.vggish_model(
            input_shape=training_config.audio_input_shape,
            input_tensor=vggish_input,
            lookback=training_config.lookback,
        )
        model_output = vggish_model(vggish_input)
        # VGGish model returns feature maps for avg/max pooling. Using LSTM for additional feature extraction.
        # Might be able to replace this with another method in the future
        return layers.Bidirectional(
            layers.LSTM(
                128,
                return_sequences=False,
                kernel_initializer=initializers.glorot_uniform(42),
            )
        )(model_output)
