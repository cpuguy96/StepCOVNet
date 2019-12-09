from scripts_common.audio_preprocessing import getMFCCBands2DMadmom, get_madmom_librosa_features
from scripts_common.parameters import *
from scripts_data_collection.sample_collection_helper import feature_onset_phrase_label_sample_weights

from os.path import join
from sklearn.preprocessing import StandardScaler

from functools import partial

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


def collect_features(audio_path,
                     annotation_path,
                     channel,
                     extra,
                     fn):
    # from the annotation to get feature, frame start and frame end of each line, frames_onset
    try:
        log_mel, frames_onset, frame_start, frame_end = \
            dump_feature_onset_helper(audio_path, annotation_path, fn, channel)

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
        return [feature, label, sample_weights, extra_feature]
    except Exception:
        print("Error collecting features for", fn)
        return [None, None, None, None]


class Worker:

    def __init__(self, multi, extra, is_limited, limit, under_sample, num_workers):
        self.pool = multiprocessing.Pool(num_workers)
        self.multi = multi
        self.extra = extra
        self.is_limited = is_limited
        self.limit = limit
        self.under_sample = under_sample

        self.features_low = []
        self.features_mid = []
        self.features_high = []
        self.features = []
        self.extra_features = []
        self.labels = []
        self.weights = []

        self.sample_count = 0
        self.song_count = 0
        self.lock = multiprocessing.Lock()

    def __collect_results(self, feature, label, sample_weight, extra_feature):
        if self.is_limited and self.sample_count >= self.limit:
            print("limit reached after %d songs. breaking..." % self.song_count)
            return False
        if feature is None or label is None or sample_weight is None or extra_feature is None:
            return True
        try:
            self.lock.acquire()
            if self.multi:
                self.features_low.append(feature[:, :, 0].astype("float16"))
                self.features_mid.append(feature[:, :, 1].astype("float16"))
                self.features_high.append(feature[:, :, 2].astype("float16"))
            else:
                self.features.append(feature)

            self.labels.append(label.astype("int8"))
            self.weights.append(sample_weight)

            if self.extra:
                self.extra_features.append(extra_feature.astype("int8"))

            if self.is_limited and self.under_sample:
                self.sample_count += label.sum()
            elif self.is_limited:
                self.sample_count += len(label)
            self.song_count += 1
            return True
        finally:
            self.lock.release()

    def __callback(self, result, use_map):
        if use_map:
            for feature, label, sample_weight, extra_feature in result:
                self.__collect_results(feature, label, sample_weight, extra_feature)
        else:
            feature, label, sample_weight, extra_feature = result
            return self.__collect_results(feature, label, sample_weight, extra_feature)

    def do_job(self, func, iter):
        print("Starting job")
        if self.limit > 0:
            with self.pool as pool:
                for result in pool.imap_unordered(func, iter):
                    if not self.__callback(result, False):
                        break
        else:
            with self.pool as pool:
                self.__callback(pool.map_async(func, iter).get(), True)


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

    if multi:
        channel = 3
    else:
        channel = 1

    if is_limited and under_sample:
        limit //= 2

    worker = Worker(multi, extra, is_limited, limit, under_sample, psutil.cpu_count(logical=False))
    worker.do_job(partial(collect_features, audio_path, annotation_path, channel, extra), getRecordings(annotation_path))

    worker.labels = np.array(np.concatenate(worker.labels, axis=0)).astype("int8")
    worker.weights = np.array(np.concatenate(worker.weights, axis=0)).astype("float16")

    if extra:
        worker.extra_features = np.array(np.concatenate(worker.extra_features, axis=0)).astype("int8")
    else:
        worker.extra_features = None

    prefix = ""

    if multi:
        prefix += "multi_"

    if under_sample:
        prefix += "under_"

    indices_used = np.asarray(range(len(worker.labels))).reshape(-1, 1)

    if under_sample:
        print("Under sampling ...")
        from imblearn.under_sampling import RandomUnderSampler
        indices_used, _ = RandomUnderSampler(random_state=42).fit_resample(indices_used, worker.labels)

    indices_used = np.sort(indices_used.reshape(-1).astype(int))

    if is_limited and len(indices_used) > limit:
        if under_sample:
            indices_used = np.sort(np.random.choice(indices_used, limit*2))
        else:
            indices_used = indices_used[:limit]

        assert worker.labels.sum() > 0, "Not enough positive labels. Increase limit!"

    print("Saving labels ...")
    joblib.dump(worker.labels[indices_used], join(path_output, prefix + 'labels.npz'), compress=True)

    print("Saving sample weights ...")
    joblib.dump(worker.weights[indices_used], join(path_output, prefix + 'sample_weights.npz'), compress=True)

    if extra:
        print("Saving extra features ...")
        joblib.dump(worker.extra_features[indices_used], join(path_output, prefix + 'extra_features.npz'), compress=True)

    if multi:
        worker.features_low = np.array(np.concatenate(worker.features_low, axis=0)[indices_used].astype("float16"))
        worker.features_mid = np.array(np.concatenate(worker.features_mid, axis=0)[indices_used].astype("float16"))
        worker.features_high = np.array(np.concatenate(worker.features_high, axis=0)[indices_used].astype("float16"))

        print("Saving multi-features ...")
        joblib.dump(np.stack([worker.features_low, worker.features_mid, worker.features_high], axis=-1).astype("float16"), join(path_output, prefix + 'dataset_features.npz'), compress=True)
        print("Saving low scaler ...")
        pickle.dump(StandardScaler().fit(worker.features_low), open(join(path_output, prefix + 'scaler_low.pkl'), 'wb'))
        print("Saving mid scaler ...")
        pickle.dump(StandardScaler().fit(worker.features_mid), open(join(path_output, prefix + 'scaler_mid.pkl'), 'wb'))
        print("Saving high scaler ...")
        pickle.dump(StandardScaler().fit(worker.features_high), open(join(path_output, prefix + 'scaler_high.pkl'), 'wb'))
    else:
        worker.features = np.array(np.concatenate(worker.features, axis=0)).astype("float16")
        print("Saving features ...")
        joblib.dump(worker.features[indices_used], join(path_output, prefix + 'dataset_features.npz'), compress=True)
        print("Saving scaler ...")
        pickle.dump(StandardScaler().fit(worker.features), open(join(path_output, prefix + 'scaler.pkl'), 'wb'))


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

    dump_feature_label_sample_weights_onset_phrase(audio_path=args.audio,
                                                   annotation_path=args.annotation,
                                                   path_output=args.output,
                                                   multi=args.multi,
                                                   extra=args.extra,
                                                   under_sample=args.under_sample,
                                                   limit=args.limit)
