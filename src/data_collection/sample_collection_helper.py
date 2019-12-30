from os.path import join

import numpy as np

from common.audio_preprocessing import get_madmom_log_mels
from configuration.parameters import SAMPLE_RATE, HOPSIZE_T


def remove_out_of_range(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, mfcc):
    frames_onset_p25 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

    len_line = frame_end - frame_start + 1

    mfcc_line = mfcc[frame_start:frame_end + 1, :]

    sample_weights = np.ones((len_line,))
    sample_weights[frames_onset_p25 - frame_start] = 0.25

    label = np.zeros((len_line,))
    label[frames_onset - frame_start] = 1
    label[frames_onset_p25 - frame_start] = 1

    return mfcc_line, label, sample_weights


def timings_parser(timing_name):
    """
    Schluter onset time annotation parser
    :param timing_name:
    :return: onset time list
    """

    with open(timing_name, 'r') as file:
        lines = [line.replace("\n", "") for line in file.readlines()]
        if "NOTES" in set(lines):
            lines = lines[lines.index("NOTES") + 1:]
            return [line.split(" ")[1] for line in lines]
        else:
            raise ValueError('Could not find NOTES line in file %s' % timing_name)


def dump_feature_onset_helper(wav_path, timing_path, file_name, multi):
    mfcc = get_madmom_log_mels(join(wav_path, file_name + '.wav'), SAMPLE_RATE, HOPSIZE_T, multi)

    times_onset = timings_parser(join(timing_path, file_name + '.txt'))
    times_onset = [float(to) for to in times_onset]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / HOPSIZE_T), dtype=int)

    # line start and end frames
    frame_start = 0
    frame_end = mfcc.shape[0] - 1

    return mfcc, frames_onset, frame_start, frame_end
