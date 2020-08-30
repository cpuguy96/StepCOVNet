import numpy as np

from stepcovnet.common.utils import get_samples_ngram_with_mask
from stepcovnet.config.InferenceConfig import InferenceConfig
from stepcovnet.data_collection.sample_collection_helper import get_audio_features
from stepcovnet.inputs.AbstractInput import AbstractInput


class InferenceInput(AbstractInput):
    def __init__(self, inference_config: InferenceConfig):
        super(InferenceInput, self).__init__(config=inference_config)
        self.audio_features = get_audio_features(wav_path=self.config.audio_path,
                                                 file_name=self.config.file_name,
                                                 config=self.config.dataset_config)
        self.arrow_input_init, self.arrow_mask_init = get_samples_ngram_with_mask(samples=np.array([0]),
                                                                                  lookback=self.config.lookback,
                                                                                  reshape=True,
                                                                                  mask_padding_value=0)
        self.arrow_input_init = self.arrow_input_init[:-1, 1:]
        self.arrow_mask_init = self.arrow_mask_init[:-1, 1:]
