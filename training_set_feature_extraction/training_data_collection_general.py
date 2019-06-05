#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import h5py
import numpy as np
from sample_collection_helper import feature_onset_phrase_label_sample_weights

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

from parameters_jingju import *
from file_path_jingju_shared import *

from audio_preprocessing import getMFCCBands2DMadmom

from utilFunctions import getRecordings


def annotationCvParser(annotation_filename):
    """
    Schluter onset time annotation parser
    :param annotation_filename:
    :return: onset time list
    """

    with open(annotation_filename, 'r') as file:
        lines = file.readlines()
        list_onset_time = [x.replace("\n", "").split(" ")[1] for x in lines[3:]]
        return list_onset_time


def feature_label_weights_saver(path_output, filename, feature, label, sample_weights):

    filename_feature_all = join(path_output, 'dataset_features.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature)
    h5f.close()

    pickle.dump(label, open(join(path_output, 'dataset_labels.pkl'), 'wb'), protocol=2)

    pickle.dump(sample_weights, open(join(path_output,  'dataset_sample_weights.pkl'), 'wb'), protocol=2)


def dump_feature_onset_helper(audio_path, annotation_path, fn, channel):

    audio_fn = join(audio_path, fn + '.wav')
    annotation_fn = join(annotation_path, fn + '.txt')

    mfcc = getMFCCBands2DMadmom(audio_fn, fs, hopsize_t, channel)

    print('Feature collecting ...', fn)

    times_onset = annotationCvParser(annotation_fn)
    times_onset = [float(to) for to in times_onset]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = 0
    frame_end = mfcc.shape[0] - 1

    return mfcc, frames_onset, frame_start, frame_end


def dump_feature_label_sample_weights_onset_phrase(audio_path, annotation_path, path_output):
    """
    dump feature, label, sample weights for each phrase with bock annotation format
    :param audio_path:
    :param annotation_path:
    :param path_output:
    :return:
    """
    for fn in getRecordings(annotation_path):

        # from the annotation to get feature, frame start and frame end of each line, frames_onset
        log_mel, frames_onset, frame_start, frame_end = dump_feature_onset_helper(audio_path, annotation_path, fn, 1)

        # simple sample weighting
        feature, label, sample_weights = \
            feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, log_mel)

        # save feature, label and weights
        feature_label_weights_saver(path_output, fn, feature, label, sample_weights)

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="dump feature, label and sample weights for general purpose.")
    parser.add_argument("--audio",
                        type=str,
                        help="input audio path")
    parser.add_argument("--annotation",
                        type=str,
                        help="input annotation path")
    parser.add_argument("--output",
                        type=str,
                        help="output path")
    args = parser.parse_args()

    if not os.path.isdir(args.audio):
        raise OSError('Audio path %s not found' % args.audio)

    if not os.path.isdir(args.annotation):
        raise OSError('Annotation path %s not found' % args.annotation)

    if not os.path.isdir(args.output):
        raise OSError('Output path %s not found' % args.output)

    dump_feature_label_sample_weights_onset_phrase(audio_path=args.audio,
                                                        annotation_path=args.annotation,
                                                        path_output=args.output)