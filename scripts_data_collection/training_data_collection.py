from scripts_common.audio_preprocessing import getMFCCBands2DMadmom, get_madmom_librosa_features
from scripts_common.parameters import *
from scripts_data_collection.sample_collection_helper import feature_onset_phrase_label_sample_weights

from os.path import join
from sklearn.preprocessing import StandardScaler

from functools import partial
from collections import namedtuple

import os
import pickle
import joblib
import psutil
import multiprocessing
import numpy as np


def getRecordings(wav_path):
    recordings      = []
    for root, subFolders, files in os.walk(wav_path):
            for f in files:
                file_prefix, file_extension = os.path.splitext(f)
                if file_prefix != '.DS_Store' and file_prefix != '_DS_Store':
                    recordings.append(file_prefix)
    return recordings


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


def dump_feature_onset_helper(audio_path, annotation_path, fn, multi):

    audio_fn = join(audio_path, fn + '.wav')
    annotation_fn = join(annotation_path, fn + '.txt')

    mfcc = getMFCCBands2DMadmom(audio_fn, fs, hopsize_t, multi)

    print('Feature collecting ...', fn)

    times_onset = annotationCvParser(annotation_fn)
    times_onset = [float(to) for to in times_onset]

    # syllable onset frames
    frames_onset = np.array(np.around(np.array(times_onset) / hopsize_t), dtype=int)

    # line start and end frames
    frame_start = 0
    frame_end = mfcc.shape[0] - 1

    return mfcc, frames_onset, frame_start, frame_end


def collect_features(audio_path,
                     annotation_path,
                     multi,
                     extra,
                     data_named_tuple,
                     fn):
    # from the annotation to get feature, frame start and frame end of each line, frames_onset
    try:
        log_mel, frames_onset, frame_start, frame_end = \
            dump_feature_onset_helper(audio_path, annotation_path, fn, multi)

        # simple sample weighting
        feature, label, sample_weights = \
            feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, log_mel)

        if extra:
            # beat frames predicted by madmom DBNBeatTrackingProcess and librosa.onset.onset_decect
            extra_feature = get_madmom_librosa_features(join(audio_path, fn + '.wav'),
                                                        fs,
                                                        hopsize_t,
                                                        len(label),
                                                        frame_start)
        else:
            extra_feature = False

        return data_named_tuple(feature, label, sample_weights, extra_feature)
    except Exception:
        print("Error collecting features for", fn)
        return None


def format_data(data,
                multi):
    labels, sample_weights, extra_features = [], [], []
    features = []
    features_low, features_mid, features_high = [], [], []

    for feature_list, label_list, sample_weight_list, extra_feature_list in data:
        if multi:
            features_low.append(feature_list[:, :, 0].astype("float16"))
            features_mid.append(feature_list[:, :, 0].astype("float16"))
            features_high.append(feature_list[:, :, 0].astype("float16"))
        else:
            features.append(feature_list)

        if extra_feature_list is not None:
            extra_features.append(extra_feature_list.astype("int8"))

        labels.append(label_list.astype("int8"))
        sample_weights.append(sample_weight_list)

    labels = np.concatenate(labels, axis=0).astype("int8")
    sample_weights = np.concatenate(sample_weights, axis=0).astype("float16")

    if not extra_features:
        extra_features = np.concatenate(extra_features, axis=0).astype("int8")

    if multi:
        features_low = np.concatenate(features_low, axis=0).astype("float16"),
        features_mid = np.concatenate(features_mid, axis=0).astype("float16"),
        features_high = np.concatenate(features_high, axis=0).astype("float16")
        all_features = [features_low, features_mid, features_high]
    else:
        all_features = np.concatenate(features, axis=0).astype("float16")

    return all_features, labels, sample_weights, extra_features


def collect_data(audio_path,
                 annotation_path,
                 multi,
                 extra,
                 is_limited,
                 limit,
                 under_sample):
    func = partial(audio_path, annotation_path, multi, extra,
                   namedtuple("AudioSampleData", ["features", "labels", "sample_weights", "extra_features"]))
    file_names = getRecordings(annotation_path)
    data = []

    with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
        if is_limited:
            sample_count = 0
            song_count = 0
            for result in pool.imap(func, file_names):
                if result is None:
                    continue
                data.append(result)
                if under_sample:
                    sample_count += result.labels.sum()  # labels collected
                else:
                    sample_count += len(result.labels)
                song_count += 1
                if sample_count >= limit:
                    print("limit reached after %d songs. breaking..." % song_count)
                    break
        else:
            data = pool.map(func, file_names)

    return format_data(data, multi)


def dump_feature_label_sample_weights_onset_phrase(audio_path,
                                                   annotation_path,
                                                   path_output,
                                                   multi,
                                                   extra,
                                                   under_sample,
                                                   limit):
    if not os.path.isdir(audio_path):
        raise NotADirectoryError('Audio path %s not found' % audio_path)

    if not os.path.isdir(annotation_path):
        raise NotADirectoryError('Annotation path %s not found' % annotation_path)

    if not os.path.isdir(path_output):
        print('Output path not found. Creating directory...')
        os.makedirs(path_output, exist_ok=True)

    if multi == 1:
        multi = True
    else:
        multi = False

    if extra == 1:
        extra = True
    else:
        extra = False

    if under_sample == 1:
        under_sample = True
    else:
        under_sample = False

    if limit == 0:
        raise ValueError('Limit cannot be 0!')

    if limit < 0:
        limit = -1
        is_limited = False
    else:
        limit = limit
        is_limited = True

    if is_limited and under_sample:
        limit //= 2

    features, labels, weights, extra_features = collect_data(audio_path,
                                                             annotation_path,
                                                             multi,
                                                             extra,
                                                             is_limited,
                                                             limit,
                                                             under_sample)

    prefix = ""

    if multi:
        prefix += "multi_"

    if under_sample:
        prefix += "under_"

    indices_used = np.asarray(range(len(labels))).reshape(-1, 1)

    if under_sample:
        print("Under sampling ...")
        from imblearn.under_sampling import RandomUnderSampler
        indices_used, _ = RandomUnderSampler(random_state=42).fit_resample(indices_used, labels)

    indices_used = np.sort(indices_used.reshape(-1).astype(int))

    if is_limited and len(indices_used) > limit:
        if under_sample:
            indices_used = np.sort(np.random.choice(indices_used, limit*2))
        else:
            indices_used = indices_used[:limit]

        assert labels.sum() > 0, "Not enough positive labels. Increase limit!"

    print("Saving labels ...")
    joblib.dump(labels[indices_used], join(path_output, prefix + 'labels.npz'), compress=True)

    print("Saving sample weights ...")
    joblib.dump(weights[indices_used], join(path_output, prefix + 'sample_weights.npz'), compress=True)

    if extra:
        print("Saving extra features ...")
        joblib.dump(extra_features[indices_used], join(path_output, prefix + 'extra_features.npz'), compress=True)

    if multi:
        print("Saving multi-features ...")
        joblib.dump(np.stack([features[0], features[1], features[2]], axis=-1).astype("float16"), join(path_output, prefix + 'dataset_features.npz'), compress=True)
        print("Saving low scaler ...")
        pickle.dump(StandardScaler().fit(features[0]), open(join(path_output, prefix + 'scaler_low.pkl'), 'wb'))
        print("Saving mid scaler ...")
        pickle.dump(StandardScaler().fit(features[1]), open(join(path_output, prefix + 'scaler_mid.pkl'), 'wb'))
        print("Saving high scaler ...")
        pickle.dump(StandardScaler().fit(features[2]), open(join(path_output, prefix + 'scaler_high.pkl'), 'wb'))
    else:
        print("Saving features ...")
        joblib.dump(features[indices_used], join(path_output, prefix + 'dataset_features.npz'), compress=True)
        print("Saving scaler ...")
        pickle.dump(StandardScaler().fit(features), open(join(path_output, prefix + 'scaler.pkl'), 'wb'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="dump feature, label and sample weights for general purpose.")
    parser.add_argument("-w", "--wav",
                        type=str,
                        help="input wavs path")
    parser.add_argument("-t", "--timing",
                        type=str,
                        help="input timings path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="output path")
    parser.add_argument("--multi",
                        type=int,
                        default=0,
                        help="whether multiple STFT window time-lengths are captured")
    parser.add_argument("--extra",
                        type=int,
                        default=0,
                        help="whether to gather extra data from madmom and librosa")
    parser.add_argument("--under_sample",
                        type=int,
                        default=0,
                        help="whether to under sample for balanced classes")
    parser.add_argument("--limit",
                        type=int,
                        default=-1,
                        help="maximum number of samples allowed to be collected")
    args = parser.parse_args()

    dump_feature_label_sample_weights_onset_phrase(audio_path=args.wav,
                                                   annotation_path=args.timing,
                                                   path_output=args.output,
                                                   multi=args.multi,
                                                   extra=args.extra,
                                                   under_sample=args.under_sample,
                                                   limit=args.limit)
