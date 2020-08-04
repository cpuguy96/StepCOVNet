class StepCOVNetModel(object):
    def __init__(self, model_path, model_name="StepCOVNet", model=None):
        self.model_path = model_path
        self.model_name = model_name
        self.model = model

    @staticmethod
    def load(model_path):
        return StepCOVNetModel(model_path=model_path)
