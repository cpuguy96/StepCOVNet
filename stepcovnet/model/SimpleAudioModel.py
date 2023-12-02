from tensorflow.keras.layers import Input, Layer

from stepcovnet import config
from stepcovnet.model.AudioModel import AudioModel


class SimpleAudioModel(AudioModel):
    def _create_audio_model(
        self, training_config: config.TrainingConfig, model_input: Input
    ) -> Layer:
        raise NotImplementedError
