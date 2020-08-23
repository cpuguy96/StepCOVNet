import numpy as np
from sklearn.preprocessing import LabelEncoder

from stepcovnet.common.constants import NUM_ARROW_TYPES
from stepcovnet.encoder.AbstractArrowEncoder import AbstractArrowEncoder


class LabelArrowEncoder(AbstractArrowEncoder):
    def __init__(self, num_note_types=NUM_ARROW_TYPES):
        encoder = LabelEncoder().fit(np.asarray(self.get_all_note_combs(num_note_types)).ravel())
        super(LabelArrowEncoder, self).__init__(encoder=encoder)

    def encode(self, arrows) -> int:
        return self.encoder.transform([arrows])[0]

    def decode(self, encoded_arrows):
        return str(self.encoder.categories_[0][encoded_arrows])

    @staticmethod
    def get_all_note_combs(num_note_types):
        all_note_combs = []

        for first_digit in range(0, num_note_types):
            for second_digit in range(0, num_note_types):
                for third_digit in range(0, num_note_types):
                    for fourth_digit in range(0, num_note_types):
                        all_note_combs.append(
                            str(first_digit) + str(second_digit) + str(third_digit) + str(fourth_digit))
        # Adding '0000' to possible note combinations.
        # This will allow the arrow prediction model to predict an empty note.
        return all_note_combs
