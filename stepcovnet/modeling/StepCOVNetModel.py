class StepCOVNetModel(object):
    def __init__(self, model_path, model_name=None, model=None):
        self.model_path = model_path
        self.model = model

    @staticmethod
    def load(model_path):
        return StepCOVNetModel(model_path=model_path)
