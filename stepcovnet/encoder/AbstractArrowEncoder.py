from abc import ABC
from abc import abstractmethod


class AbstractArrowEncoder(ABC, object):
    def __init__(self, encoder):
        self.encoder = encoder

    @abstractmethod
    def encode(self, arrows):
        ...

    @abstractmethod
    def decode(self, encoded_arrows) -> str:
        ...
