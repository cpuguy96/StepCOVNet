from abc import ABC
from abc import abstractmethod

from stepcovnet.common.tf_config import tf_init
from stepcovnet.modeling.StepCOVNetModel import StepCOVNetModel


class AbstractExecutor(ABC, object):
    def __init__(self, stepcovnet_model: StepCOVNetModel, *args, **kwargs):
        self.stepcovnet_model = stepcovnet_model
        tf_init()

    @abstractmethod
    def execute(self, input_data):
        pass
