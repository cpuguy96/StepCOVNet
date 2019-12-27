import os
from os.path import join

import numpy as np

from common.audio_preprocessing import get_madmom_log_mels
from configuration.parameters import sample_rate, hopsize_t


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


def get_recordings(wav_path):
    recordings = []
    for root, subFolders, files in os.walk(wav_path):
        for f in files:
            file_prefix, file_extension = os.path.splitext(f)
            if file_prefix != '.DS_Store' and file_prefix != '_DS_Store':
                recordings.append(file_prefix)
    return recordings


def timings_parser(annotation_filename):
    """
    Schluter onset time annotation parser
    :param annotation_filename:
    :return: onset time list
    """

    with open(annotation_filename, 'r') as file:
        lines = file.readlines()
        list_onset_time = [x.replace("\n", "").split(" ")[1] for x in lines[3:]]
    return list_onset_time


def dump_feature_onset_helper(audio_path, annotation_path, fn, multi):
    audio_fn = join(audio_path, fn + '.wav')
    annotation_fn = join(annotation_path, fn + '.txt')

    mfcc = get_madmom_log_mels(audio_fn, sample_rate, hopsize_t, multi)

    print('Feature collecting ...', fn)

    times_onset = timings_parser(annotation_fn)
    times_onset = [float(to) for to in times_onset]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = 0
    frame_end = mfcc.shape[0] - 1

    return mfcc, frames_onset, frame_start, frame_end
