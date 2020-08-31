from stepcovnet.config.AbstractConfig import AbstractConfig


class InferenceConfig(AbstractConfig):
    def __init__(self, audio_path, file_name, dataset_config, lookback, difficulty, scalers=None):
        super(InferenceConfig, self).__init__(dataset_config=dataset_config, lookback=lookback, difficulty=difficulty)
        self.audio_path = audio_path
        self.file_name = file_name
        self.scalers = scalers
