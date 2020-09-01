import numpy as np
from sklearn.preprocessing import LabelEncoder

from stepcovnet.common.constants import ALL_ARROW_COMBS
from stepcovnet.encoder.AbstractArrowEncoder import AbstractArrowEncoder


class LabelArrowEncoder(AbstractArrowEncoder):
    def __init__(self, all_arrow_combs: np.array = ALL_ARROW_COMBS):
        encoder = LabelEncoder().fit(all_arrow_combs.ravel())
        super(LabelArrowEncoder, self).__init__(encoder=encoder)

    def encode(self, arrows) -> np.array:
        return self.encoder.transform([arrows])[0]

    def decode(self, encoded_arrows):
        return str(self.encoder.categories_[0][encoded_arrows])
