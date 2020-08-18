from stepcovnet.config.AbstractConfig import AbstractConfig


class InferenceConfig(AbstractConfig):
    def __init__(self, audio_path, dataset_config, lookback):
        super(InferenceConfig, self).__init__()
        self.audio_path = audio_path
        self.dataset_config = dataset_config
        self.lookback = lookback
        self.sample_frequency = self.dataset_config["SAMPLE_RATE"]
