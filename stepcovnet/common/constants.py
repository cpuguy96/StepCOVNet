import numpy as np

SAMPLE_RATE = 44100

HOPSIZE_T = 0.010

MULTI_CHANNEL_FRAME_SIZES = [1024, 2048, 4096]

SINGLE_CHANNEL_FRAME_SIZE = 2048

NUM_MULTI_CHANNELS = len(MULTI_CHANNEL_FRAME_SIZES)

NUM_FREQ_BANDS = 80

NUM_TIME_BANDS = 15

THRESHOLDS = {'expert': 0.5}

NUM_ARROW_TYPES = 4  # TODO: Move this to dataset config

ARROW_NAMES = ["left_arrow", "down_arrow", "up_arrow", "right_arrow"]

NUM_ARROWS = len(ARROW_NAMES)


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


ALL_ARROW_COMBS = np.array(get_all_note_combs(NUM_ARROW_TYPES))

NUM_ARROW_COMBS = len(ALL_ARROW_COMBS)
