from stepcovnet.executor.AbstractExecutor import AbstractExecutor
from stepcovnet.model_input.InferenceInput import InferenceInput


class InferenceExecutor(AbstractExecutor):
    def __init__(self, inference_input: InferenceInput, stepcovnet_model):
        super(InferenceExecutor, self).__init__(input_data=inference_input, stepcovnet_model=stepcovnet_model)

    def execute(self):
        pass
