import numpy as np


class BinaryArrowEncoder(object):
    def __init__(self):
        pass

    def encode(self, arrow):
        return np.array([0, 0, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 0,
                         0, 0, 0, 0], dtype=int)

    def decode(self, encoded_arrow):
        pass
