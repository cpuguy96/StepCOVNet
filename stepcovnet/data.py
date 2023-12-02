from enum import Enum

from transformers import GPT2Tokenizer

from stepcovnet.dataset.DistributedModelDataset import DistributedModelDataset
from stepcovnet.dataset.ModelDataset import ModelDataset


class Tokenizers(Enum):
    GPT2 = GPT2Tokenizer.from_pretrained("gpt2")


class ModelDatasetTypes(Enum):
    SINGULAR_DATASET = ModelDataset
    DISTRIBUTED_DATASET = DistributedModelDataset
