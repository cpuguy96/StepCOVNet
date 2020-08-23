import numpy as np

from stepcovnet.common.constants import NUM_ARROWS
from stepcovnet.common.constants import NUM_ARROW_TYPES
from stepcovnet.encoder.BinaryArrowEncoder import BinaryArrowEncoder
from stepcovnet.encoder.LabelArrowEncoder import LabelArrowEncoder
from stepcovnet.executor.AbstractExecutor import AbstractExecutor
from stepcovnet.model_input.InferenceInput import InferenceInput


class InferenceExecutor(AbstractExecutor):
    def __init__(self, stepcovnet_model):
        super(InferenceExecutor, self).__init__(stepcovnet_model=stepcovnet_model)
        self.binary_arrow_encoder = BinaryArrowEncoder()
        self.label_arrow_encoder = LabelArrowEncoder()

    def execute(self, input_data: InferenceInput):
        arrow_input = input_data.arrow_input_init
        arrow_mask = input_data.arrow_mask_init

        pred_arrows = []
        for audio_input in input_data.audio_input:
            binary_arrows_probs = self.stepcovnet_model.model.predict({
                "arrow_input": arrow_input,
                "arrow_mask": arrow_mask,
                "audio_input": audio_input,
            })[0]  # Index by 0 since predictions are returned in a batch
            binary_encoded_arrows = np.array([])
            for i in range(NUM_ARROWS):
                binary_arrow_prob = binary_arrows_probs[NUM_ARROW_TYPES * i: NUM_ARROW_TYPES * (i + 1)]
                encoded_arrows = np.random.choice(NUM_ARROW_TYPES, 1, p=binary_arrow_prob)[0]
                binary_encoded_arrows = np.append(binary_encoded_arrows, encoded_arrows)
            arrows = self.binary_arrow_encoder.decode(encoded_arrows=binary_encoded_arrows)
            pred_arrows.append(arrows)
            # Roll and append predicted arrow to input to predict next sample
            arrow_input = np.roll(arrow_input, -1, axis=0)
            arrow_mask = np.roll(arrow_mask, -1, axis=0)
            arrow_input[0][-1] = self.label_arrow_encoder.encode(arrows)
            arrow_mask[0][-1] = 0
        return pred_arrows
