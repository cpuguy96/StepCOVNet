from abc import ABC


class AbstractInput(ABC, object):
    def __init__(self, config, *args, **kwargs):
        self.config = config
