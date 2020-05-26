from collections import defaultdict
from os.path import join

import numpy as np

from stepcovnet.common.audio_preprocessing import get_madmom_log_mels
from stepcovnet.common.parameters import HOPSIZE_T
from stepcovnet.common.parameters import SAMPLE_RATE
from stepcovnet.common.utils import feature_reshape
from stepcovnet.common.utils import get_arrow_label_encoder


def remove_out_of_range(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def feature_onset_phrase_label_sample_weights(frames_onset, mfcc, arrows):
    frame_start = 0
    frame_end = mfcc.shape[0] - 1
    labels_dict = defaultdict(np.array)
    sample_weights_dict = defaultdict(np.array)
    arrows_dict = defaultdict(np.array)

    for difficulty, onsets in frames_onset.items():
        frames_onsets_p25 = np.hstack((onsets - 1, onsets + 1))
        frames_onsets_p25 = np.sort(remove_out_of_range(frames_onsets_p25, frame_start, frame_end))

        len_line = frame_end - frame_start + 1

        sample_weights = np.ones((len_line,))
        sample_weights[frames_onsets_p25 - frame_start] = 0.25
        sample_weights_dict[difficulty] = sample_weights.astype("float16")

        label = np.zeros((len_line,))
        label[onsets - frame_start] = 1
        label[frames_onsets_p25 - frame_start] = 1
        labels_dict[difficulty] = label.astype("int8")

        arrows_array = np.zeros((len_line,))
        arrows_list = arrows[difficulty].reshape(-1)
        i = 0
        for onset, arrow in zip(onsets, arrows_list):
            arrows_array[onset - frame_start] = arrow
            # This should be fine since timings should never be right next to each other
            if 2 * i < len(frames_onsets_p25):
                arrows_array[frames_onsets_p25[2 * i] - frame_start] = arrow
            if 2 * i + 1 < len(frames_onsets_p25):
                arrows_array[frames_onsets_p25[2 * i + 1] - frame_start] = arrow
            i += 1
        arrows_dict[difficulty] = arrows_array.astype("int32")

    mfcc_line = mfcc[frame_start:frame_end + 1, :]

    return mfcc_line, labels_dict, sample_weights_dict, arrows_dict


def timings_parser(timing_name):
    """
    Read each line of timings file and parse arrows and timings
    :param timing_name: str - file name containing note timings
    :return: defaultdict - key: difficulty; value: list containing note arrows and timings
    """

    with open(timing_name, 'r') as file:
        data = defaultdict(dict)
        read_timings = False
        curr_difficulty = None
        encoder = get_arrow_label_encoder()
        for line in file.readlines():
            line = line.replace("\n", "")
            if line.startswith("NOTES"):
                read_timings = True
            elif read_timings:
                if line.startswith("DIFFICULTY"):
                    new_difficulty = line.split()[1].lower()
                    if new_difficulty in data:
                        raise ValueError("Same difficulty detected in the song data.")
                    curr_difficulty = new_difficulty
                elif curr_difficulty is not None:
                    arrow, timing = line.split(" ")[0:2]
                    data[curr_difficulty][float(timing)] = encoder.transform([arrow])
        return data


def dump_feature_onset_helper(wav_path, note_data_path, file_name, multi):
    mfcc = get_madmom_log_mels(join(wav_path, file_name + '.wav'), SAMPLE_RATE, HOPSIZE_T, multi)
    mfcc = feature_reshape(mfcc, multi)

    note_data = timings_parser(join(note_data_path, file_name + '.txt'))

    # convert note timings into frame timings
    frames_onset = defaultdict(np.array)
    arrows_dict = defaultdict(np.array)

    for difficulty, data in note_data.items():
        timings, arrows = list(data.keys()), list(data.values())
        frames_onset[difficulty] = np.array(np.around(np.array(timings) / HOPSIZE_T), dtype=int)
        arrows_dict[difficulty] = np.array(arrows, dtype=int)

    return mfcc, frames_onset, arrows_dict
