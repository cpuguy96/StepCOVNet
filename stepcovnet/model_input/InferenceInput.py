from stepcovnet.model_input.AbstractInput import AbstractInput


class InferenceInput(AbstractInput):
    def __init__(self, inference_config):
        super(InferenceInput, self).__init__(config=inference_config)
