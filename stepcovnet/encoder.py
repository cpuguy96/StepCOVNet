from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from stepcovnet.common.constants import NUM_ARROW_TYPES, NUM_ARROWS, ALL_ARROW_COMBS


class AbstractArrowEncoder(ABC, object):
    def __init__(self, encoder):
        self.encoder = encoder

    @abstractmethod
    def encode(self, arrows):
        ...

    @abstractmethod
    def decode(self, encoded_arrows) -> str:
        ...


class BinaryArrowEncoder(AbstractArrowEncoder):
    def __init__(self, num_arrow_types=NUM_ARROW_TYPES):
        self.num_arrow_types = num_arrow_types
        encoder = OneHotEncoder(categories="auto", sparse=False).fit(
            np.arange(num_arrow_types).reshape(-1, 1)
        )
        super(BinaryArrowEncoder, self).__init__(encoder=encoder)

    def encode(self, arrows: Sequence[Any]) -> np.ndarray:
        return np.append(
            [],
            [
                self.encoder.transform(np.array([int(arrow)]).reshape(-1, 1))[0]
                for arrow in arrows
            ],
        ).astype(int)

    def decode(self, encoded_arrows) -> str:
        if len(encoded_arrows) / self.num_arrow_types != 4:
            raise ValueError(
                "Number of arrow types does not match encoded arrow input "
                "(%d arrow types, %d encoded arrow bits)"
                % (self.num_arrow_types, len(encoded_arrows))
            )
        arrows = []
        for i in range(NUM_ARROWS):
            encoded_arrow = encoded_arrows[
                self.num_arrow_types * i : self.num_arrow_types * (i + 1)
            ].astype(int)
            arrow = self.encoder.categories_[0][encoded_arrow]
            arrows.append(str(arrow))
        return "".join(arrows)


class LabelArrowEncoder(AbstractArrowEncoder):
    def __init__(self, all_arrow_combs: np.array = ALL_ARROW_COMBS):
        encoder = LabelEncoder().fit(all_arrow_combs.ravel())
        super(LabelArrowEncoder, self).__init__(encoder=encoder)

    def encode(self, arrows: str) -> np.ndarray:
        return self.encoder.transform([arrows])[0]

    def decode(self, encoded_arrows) -> str:
        return str(self.encoder.inverse_transform([encoded_arrows])[0])


class OneHotArrowEncoder(AbstractArrowEncoder):
    def __init__(self, all_arrow_combs: np.array = ALL_ARROW_COMBS):
        encoder = OneHotEncoder(categories="auto", sparse=False).fit(
            all_arrow_combs.reshape(-1, 1)
        )
        super(OneHotArrowEncoder, self).__init__(encoder=encoder)

    def encode(self, arrows) -> int:
        data = np.array([arrows]).reshape(1, -1)
        return self.encoder.transform(data)[0]

    def decode(self, encoded_arrows) -> str:
        return str(self.encoder.categories_[0][encoded_arrows])
