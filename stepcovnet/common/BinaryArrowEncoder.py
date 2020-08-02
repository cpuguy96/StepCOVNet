import numpy as np
from sklearn.preprocessing import OneHotEncoder


class BinaryArrowEncoder(object):
    def __init__(self, num_arrow_types=4):
        self.encoder = OneHotEncoder(categories='auto', sparse=False).fit(np.arange(num_arrow_types).reshape(-1, 1))

    def encode(self, arrows):
        return np.append([], [self.encoder.transform(np.array([int(arrow)]).reshape(-1, 1))[0] for arrow in arrows]) \
            .astype(int)

    def decode(self, encoded_arrow):
        pass
