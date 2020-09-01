from enum import Enum

from transformers import GPT2Tokenizer


class Tokenizers(Enum):
    GPT2 = GPT2Tokenizer.from_pretrained('gpt2')
