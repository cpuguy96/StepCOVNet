import json
import os
from datetime import datetime

from keras import models

from stepcovnet.config.TrainingConfig import TrainingConfig


class StepCOVNetModel(object):
    def __init__(
        self,
        model_root_path: str,
        model_name: str = "StepCOVNet",
        model: models.Model = None,
        metadata: dict = None,
    ):
        self.model_root_path = os.path.abspath(model_root_path)
        self.model_name = model_name
        self.model: models.Model = model
        self.metadata = metadata

    def build_metadata_from_training_config(
        self, training_config: TrainingConfig
    ) -> dict:
        self.metadata = {
            "model_name": self.model_name,
            "creation_time": datetime.utcnow().strftime("%b %d %Y %H:%M:%S UTC"),
            "training_config": {
                "limit": training_config.limit,
                "lookback": training_config.lookback,
                "difficulty": training_config.difficulty,
                "tokenizer_name": training_config.tokenizer_name,
                "hyperparameters": str(training_config.hyperparameters),
            },
            "dataset_config": training_config.dataset_config,
        }
        return self.metadata

    @classmethod
    def load(
        cls, input_path: str, retrained: bool = False, compile_model: bool = False
    ):
        metadata = json.load(open(os.path.join(input_path, "metadata.json"), "r"))
        model_name = metadata["model_name"]
        model_path = (
            os.path.join(input_path, model_name + "_retrained")
            if retrained
            else os.path.join(input_path, model_name)
        )
        model = models.load_model(model_path, compile=compile_model)
        return cls(
            model_root_path=input_path,
            model_name=model_name,
            model=model,
            metadata=metadata,
        )
