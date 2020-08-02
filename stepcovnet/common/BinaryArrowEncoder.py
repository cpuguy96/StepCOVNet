import numpy as np
from sklearn.preprocessing import OneHotEncoder

from stepcovnet.common.parameters import NUM_ARROW_TYPES


class BinaryArrowEncoder(object):
    def __init__(self):
        self.encoder = OneHotEncoder(categories='auto', sparse=False).fit(np.arange(NUM_ARROW_TYPES).reshape(-1, 1))

    def encode(self, arrows):
        return np.append([], [self.encoder.transform(np.array([int(arrow)]).reshape(-1, 1))[0] for arrow in arrows]) \
            .astype(int)

    def decode(self, encoded_arrow):
        pass
