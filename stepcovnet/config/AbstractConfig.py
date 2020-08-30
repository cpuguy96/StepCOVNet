from abc import ABC

from stepcovnet.common.constants import NUM_ARROWS
from stepcovnet.common.constants import NUM_ARROW_TYPES


class AbstractConfig(ABC, object):
    def __init__(self, dataset_config, lookback, difficulty, *args, **kwargs):
        self.dataset_config = dataset_config
        self.lookback = lookback
        self.difficulty = difficulty

    @property
    def arrow_input_shape(self):
        return (None,)

    @property
    def arrow_mask_shape(self):
        return (None,)

    @property
    def audio_input_shape(self):
        return self.lookback, self.dataset_config["NUM_TIME_BANDS"], self.dataset_config["NUM_FREQ_BANDS"], 1,

    @property
    def label_shape(self):
        return (NUM_ARROWS * NUM_ARROW_TYPES,)
