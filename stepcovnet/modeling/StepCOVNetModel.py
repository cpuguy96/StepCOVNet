import json
import os
from datetime import datetime

from tensorflow.keras.models import load_model


class StepCOVNetModel(object):
    def __init__(self, model_path, model_name="StepCOVNet", model=None, metadata=None):
        self.model_path = os.path.abspath(model_path)
        self.model_name = model_name
        self.model = model
        self.metadata = metadata

    def build_metadata_from_training_config(self, training_config):
        self.metadata = {
            "model_name": self.model_name,
            "creation_time": datetime.utcnow().strftime("%b %d %Y %H:%M:%S UTC"),
            "training_config": {
                "limit": training_config.limit,
                "lookback": training_config.lookback,
                "difficulty": training_config.difficulty,
                "hyperparameters": str(training_config.hyperparameters)
            },
            "dataset_config": training_config.dataset_config
        }

    @staticmethod
    def load(input_path, retrained=False):
        metadata = json.load(open(os.path.join(input_path, "metadata.json"), 'r'))
        model_name = metadata["model_name"]
        model_path = os.path.join(input_path, model_name + '_retrained') if retrained \
            else os.path.join(input_path, model_name)
        model = load_model(model_path, compile=False)
        return StepCOVNetModel(model_path=input_path, model_name=model_name, model=model, metadata=metadata)
