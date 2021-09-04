import tensorflow as tf
from tensorflow.keras.layers import Input

from stepcovnet.config.TrainingConfig import TrainingConfig
from stepcovnet.model.AudioModel import AudioModel


class SimpleAudioModel(AudioModel):
    def _create_audio_model(self, training_config: TrainingConfig, model_input: Input,
                            architecture) -> tf.keras.layers.Layer:
        raise NotImplementedError
