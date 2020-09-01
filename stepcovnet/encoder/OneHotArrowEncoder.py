import numpy as np
from sklearn.preprocessing import OneHotEncoder

from stepcovnet.common.constants import ALL_ARROW_COMBS
from stepcovnet.encoder.AbstractArrowEncoder import AbstractArrowEncoder


class OneHotArrowEncoder(AbstractArrowEncoder):
    def __init__(self, all_arrow_combs: np.array = ALL_ARROW_COMBS):
        encoder = OneHotEncoder(categories='auto', sparse=False).fit(all_arrow_combs.reshape(-1, 1))
        super(OneHotArrowEncoder, self).__init__(encoder=encoder)

    def encode(self, arrows) -> int:
        data = np.array([arrows]).reshape(1, -1)
        return self.encoder.transform(data)[0]

    def decode(self, encoded_arrows):
        return str(self.encoder.categories_[0][encoded_arrows])
