from abc import ABC
from abc import abstractmethod

from stepcovnet.common.tf_config import tf_init
from stepcovnet.modeling.StepCOVNetModel import StepCOVNetModel


class AbstractExecutor(ABC):
    def __init__(self, input_data, stepcovnet_model: StepCOVNetModel, *args, **kwargs):
        self.input_data = input_data
        self.stepcovnet_model = stepcovnet_model
        tf_init()

    @abstractmethod
    def execute(self):
        pass
