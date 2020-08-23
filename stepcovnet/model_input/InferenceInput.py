import numpy as np

from stepcovnet.common.utils import get_samples_ngram_with_mask
from stepcovnet.config.InferenceConfig import InferenceConfig
from stepcovnet.data_collection.sample_collection_helper import get_audio_features
from stepcovnet.model_input.AbstractInput import AbstractInput


class InferenceInput(AbstractInput):
    def __init__(self, inference_config: InferenceConfig):
        super(InferenceInput, self).__init__(config=inference_config)
        self.audio_features = get_audio_features(wav_path=self.config.audio_path,
                                                 file_name=self.config.file_name,
                                                 config=self.config.dataset_config)
        self.audio_input, _ = get_samples_ngram_with_mask(samples=self.audio_features,
                                                          lookback=self.config.lookback,
                                                          squeeze=False)
        self.arrow_input_init, self.arrow_mask_init = get_samples_ngram_with_mask(samples=np.array([0]),
                                                                                  lookback=self.config.lookback,
                                                                                  reshape=True)
        # Lookback data from ngram returns empty value in index 0. Also, arrow features should only
        # contain previously seen features. Therefore, removing last element and last lookback from
        # arrows features and first element from audio features.
        self.audio_input = self.audio_input[1:]
        self.arrow_input_init = self.arrow_input_init[:-1, 1:]
        self.arrow_mask_init = self.arrow_mask_init[:-1, 1:]
