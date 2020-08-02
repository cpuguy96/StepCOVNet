from enum import Enum

from stepcovnet.dataset.DistributedModelDataset import DistributedModelDataset
from stepcovnet.dataset.ModelDataset import ModelDataset


class ModelDatasetTypes(Enum):
    SINGULAR_DATASET = ModelDataset
    DISTRIBUTED_DATASET = DistributedModelDataset
