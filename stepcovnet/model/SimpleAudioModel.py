from keras import layers

from stepcovnet.config.TrainingConfig import TrainingConfig
from stepcovnet.model.AudioModel import AudioModel


class SimpleAudioModel(AudioModel):
    def _create_audio_model(
        self, training_config: TrainingConfig, model_input: layers.Input
    ) -> layers.Layer:
        raise NotImplementedError
