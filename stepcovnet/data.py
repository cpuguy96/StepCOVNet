from enum import Enum

from transformers import GPT2Tokenizer

from stepcovnet import dataset


class Tokenizers(Enum):
    GPT2 = GPT2Tokenizer.from_pretrained("gpt2")


class ModelDatasetTypes(Enum):
    SINGULAR_DATASET = dataset.ModelDataset
    DISTRIBUTED_DATASET = dataset.DistributedModelDataset
