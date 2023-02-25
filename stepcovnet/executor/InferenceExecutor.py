import numpy as np
import tensorflow as tf

from stepcovnet.common.constants import NUM_ARROW_COMBS
from stepcovnet.common.utils import apply_scalers
from stepcovnet.common.utils import get_samples_ngram_with_mask
from stepcovnet.encoder.BinaryArrowEncoder import BinaryArrowEncoder
from stepcovnet.encoder.LabelArrowEncoder import LabelArrowEncoder
from stepcovnet.encoder.OneHotArrowEncoder import OneHotArrowEncoder
from stepcovnet.executor.AbstractExecutor import AbstractExecutor
from stepcovnet.inputs.InferenceInput import InferenceInput


class InferenceExecutor(AbstractExecutor):
    def __init__(self, stepcovnet_model, verbose=False):
        super(InferenceExecutor, self).__init__(stepcovnet_model=stepcovnet_model)
        self.verbose = verbose
        self.binary_arrow_encoder = BinaryArrowEncoder()
        self.label_arrow_encoder = LabelArrowEncoder()
        self.onehot_arrow_encoder = OneHotArrowEncoder()

    def execute(self, input_data: InferenceInput):
        arrow_input = input_data.arrow_input_init
        arrow_mask = input_data.arrow_mask_init
        pred_arrows = []
        inferer = self.stepcovnet_model.model.signatures["serving_default"]
        for audio_features_index in range(len(input_data.audio_features)):
            audio_features = get_samples_ngram_with_mask(
                samples=input_data.audio_features[
                        max(audio_features_index + 1 - input_data.config.lookback, 0): audio_features_index + 1],
                lookback=input_data.config.lookback,
                squeeze=False)[0][-1]
            audio_input = apply_scalers(features=audio_features, scalers=input_data.config.scalers)
            binary_arrows_probs = inferer(arrow_input=tf.convert_to_tensor(arrow_input),
                                          arrow_mask=tf.convert_to_tensor(arrow_mask),
                                          audio_input=tf.convert_to_tensor(audio_input))
            arrows_probs = next(iter(binary_arrows_probs.values())).numpy().ravel()
            encoded_arrows = np.random.choice(NUM_ARROW_COMBS, 1, p=arrows_probs)[0]
            arrows = self.onehot_arrow_encoder.decode(encoded_arrows)
            pred_arrows.append(arrows)
            # Roll and append predicted arrow to input to predict next sample
            arrow_input = np.roll(arrow_input, -1, axis=0)
            arrow_mask = np.roll(arrow_mask, -1, axis=0)
            arrow_input[0][-1] = self.label_arrow_encoder.encode(arrows)
            arrow_mask[0][-1] = 1
            if self.verbose and audio_features_index % 100 == 0:
                print("[%d/%d] Samples generated" % (audio_features_index, len(input_data.audio_features)))
        return pred_arrows
