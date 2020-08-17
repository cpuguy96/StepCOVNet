from stepcovnet.executor.AbstractExecutor import AbstractExecutor


class InferenceExecutor(AbstractExecutor):
    def __init__(self, input_data, stepcovnet_model):
        super(InferenceExecutor, self).__init__(input_data=input_data, stepcovnet_model=stepcovnet_model)

    def execute(self):
        pass
