from abc import ABC


class AbstractInput(ABC):
    def __init__(self, config, *args, **kwargs):
        self.config = config
