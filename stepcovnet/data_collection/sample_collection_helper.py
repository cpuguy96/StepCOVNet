from collections import defaultdict
from os.path import join

import numpy as np

from stepcovnet.common.audio_preprocessing import get_madmom_log_mels
from stepcovnet.common.parameters import HOPSIZE_T
from stepcovnet.common.parameters import SAMPLE_RATE
from stepcovnet.common.utils import feature_reshape


def remove_out_of_range(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def feature_onset_phrase_label_sample_weights(frames_onset, mfcc):
    frame_start = 0
    frame_end = mfcc.shape[0] - 1
    labels_dict = defaultdict(np.array)
    sample_weights_dict = defaultdict(np.array)

    for key, value in frames_onset.items():
        frames_onset_p25 = np.hstack((value - 1, value + 1))
        frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

        len_line = frame_end - frame_start + 1

        sample_weights = np.ones((len_line,))
        sample_weights[frames_onset_p25 - frame_start] = 0.25
        sample_weights_dict[key] = sample_weights.astype("float16")

        label = np.zeros((len_line,))
        label[value - frame_start] = 1
        label[frames_onset_p25 - frame_start] = 1
        labels_dict[key] = label.astype("int8")

    mfcc_line = mfcc[frame_start:frame_end + 1, :]

    return mfcc_line, labels_dict, sample_weights_dict


def timings_parser(timing_name):
    """
    Read each line of note timings
    :param timing_name: str - file name containing note timings
    :return: defaultdict - key: timings difficulty; value: list containing note timings
    """

    with open(timing_name, 'r') as file:
        timings = defaultdict(list)
        read_timings = False
        curr_difficulty = None
        for line in file.readlines():
            line = line.replace("\n", "")
            if line.startswith("NOTES"):
                read_timings = True
            elif read_timings:
                if line.startswith("DIFFICULTY"):
                    curr_difficulty = line.split()[1].lower()
                elif curr_difficulty is not None:
                    timings[curr_difficulty].append(float(line.split(" ")[1]))
        return timings


def dump_feature_onset_helper(wav_path, timing_path, file_name, multi):
    mfcc = get_madmom_log_mels(join(wav_path, file_name + '.wav'), SAMPLE_RATE, HOPSIZE_T, multi)
    mfcc = feature_reshape(mfcc, multi)

    times_onset = timings_parser(join(timing_path, file_name + '.txt'))

    # convert note timings into frame timings
    frames_onset = defaultdict(np.array)
    for key, value in times_onset.items():
        frames_onset[key] = np.array(np.around(np.array(value) / HOPSIZE_T), dtype=int)

    # line start and end frames

    return mfcc, frames_onset
