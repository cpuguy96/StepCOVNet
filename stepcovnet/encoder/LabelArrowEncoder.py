import numpy as np
from sklearn.preprocessing import LabelEncoder

from stepcovnet.common.constants import ALL_ARROW_COMBS
from stepcovnet.encoder.AbstractArrowEncoder import AbstractArrowEncoder


class LabelArrowEncoder(AbstractArrowEncoder):
    def __init__(self, all_arrow_combs: np.array = ALL_ARROW_COMBS):
        encoder = LabelEncoder().fit(all_arrow_combs.ravel())
        super(LabelArrowEncoder, self).__init__(encoder=encoder)

    def encode(self, arrows: str) -> np.ndarray:
        return self.encoder.transform([arrows])[0]

    def decode(self, encoded_arrows) -> str:
        return str(self.encoder.inverse_transform([encoded_arrows])[0])
