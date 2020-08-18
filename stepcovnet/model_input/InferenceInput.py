from stepcovnet.config.InferenceConfig import InferenceConfig
from stepcovnet.model_input.AbstractInput import AbstractInput


class InferenceInput(AbstractInput):
    def __init__(self, inference_config: InferenceConfig):
        super(InferenceInput, self).__init__(config=inference_config)
