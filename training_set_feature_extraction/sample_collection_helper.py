import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from parameters import *


def remove_out_of_range(frames, frame_start, frame_end):
    return frames[np.all([frames <= frame_end, frames >= frame_start], axis=0)]


def feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, mfcc):
    frames_onset_p25 = np.hstack((frames_onset - 1, frames_onset + 1))
    frames_onset_p25 = remove_out_of_range(frames_onset_p25, frame_start, frame_end)

    len_line = frame_end - frame_start + 1

    mfcc_line = mfcc[frame_start:frame_end+1, :]

    sample_weights = np.ones((len_line,))
    sample_weights[frames_onset_p25 - frame_start] = 0.25

    label = np.zeros((len_line,))
    label[frames_onset - frame_start] = 1
    label[frames_onset_p25 - frame_start] = 1

    return mfcc_line, label, sample_weights


def get_onset_in_frame_helper(recording_name, idx, lab, u_list):
    """
    retrieve onset time of the syllable from textgrid
    :param recording_name:
    :param idx:
    :param lab:
    :param u_list:
    :return:
    """
    print ('Processing feature collecting ... ' + recording_name + ' phrase ' + str(idx + 1))

    if not lab:
        times_onset = [u[0] for u in u_list[1]]
    else:
        times_onset = [u[0] for u in u_list]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = frames_onset[0]

    if not lab:
        frame_end = int(u_list[0][1] / hopsize_t)
    else:
        frame_end = int(u_list[-1][1] / hopsize_t)

    return frames_onset, frame_start, frame_end

