import multiprocessing
import os
import pickle
from functools import partial
from os.path import join

import joblib
import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler

from common.audio_preprocessing import get_madmom_librosa_features
from common.utils import get_filenames_from_folder, get_filename
from configuration.parameters import HOPSIZE_T, SAMPLE_RATE
from data_collection.sample_collection_helper import feature_onset_phrase_label_sample_weights, \
    dump_feature_onset_helper


def collect_features(wav_path, timing_path, multi, extra, file_name):
    # from the annotation to get feature, frame start and frame end of each line, frames_onset
    try:
        print('Feature collecting: %s' % file_name)
        log_mel, frames_onset, frame_start, frame_end = \
            dump_feature_onset_helper(wav_path, timing_path, file_name, multi)

        # simple sample weighting
        feature, label, sample_weights = \
            feature_onset_phrase_label_sample_weights(frames_onset, frame_start, frame_end, log_mel)

        if extra:
            # beat frames predicted by madmom DBNBeatTrackingProcess and librosa.onset.onset_decect
            extra_feature = get_madmom_librosa_features(join(wav_path, file_name + '.wav'),
                                                        SAMPLE_RATE,
                                                        HOPSIZE_T,
                                                        len(label),
                                                        frame_start)
        else:
            extra_feature = None

        return [feature.astype("float16"), label.astype("int8"), sample_weights.astype("float16"), extra_feature]
    except Exception as ex:
        print("Error collecting features for %s: %r" % (file_name, ex))
        return None


def format_data(features, labels, weights, extra_features, data):
    feature_list, label_list, sample_weight_list, extra_feature_list = data

    features.extend(feature_list)
    labels.extend(label_list)
    weights.extend(sample_weight_list)
    if extra_feature_list is not None:
        extra_features.extend(extra_feature_list)

    return features, labels, weights, extra_features


def collect_data(wavs_path, timings_path, multi, extra, limit, under_sample):
    func = partial(collect_features, wavs_path, timings_path, multi, extra)
    file_names = [get_filename(file_name) for file_name in get_filenames_from_folder(timings_path)]

    features, labels, weights, extra_features = [], [], [], []

    with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
        sample_count = 0
        song_count = 0
        for result in pool.imap_unordered(func, file_names):
            if result is None:
                continue
            features, labels, weights, extra_features = format_data(features, labels, weights, extra_features, result)
            if limit > 0:
                sample_count += result[2].sum() if under_sample else len(result[2])  # labels collected
                song_count += 1
                if sample_count >= limit:
                    print("Limit reached after %d songs. Breaking..." % song_count)
                    break
    return features, labels, weights, extra_features


def training_data_collection(wavs_path, timings_path, output_path, multi_int, extra_int, under_sample_int, limit):
    if not os.path.isdir(wavs_path):
        raise NotADirectoryError('Audio path %s not found' % wavs_path)

    if not os.path.isdir(timings_path):
        raise NotADirectoryError('Annotation path %s not found' % timings_path)

    if not os.path.isdir(output_path):
        print('Output path not found. Creating directory...')
        os.makedirs(output_path, exist_ok=True)

    if limit == 0:
        raise ValueError('Limit cannot be 0!')

    multi = True if multi_int == 1 else False
    extra = True if extra_int == 1 else False
    under_sample = True if under_sample_int == 1 else False
    if limit < 0:
        limit = -1
    else:
        limit = limit

    if limit > 0 and under_sample:
        limit //= 2

    features, labels, weights, extra_features = collect_data(wavs_path, timings_path, multi, extra, limit, under_sample)

    prefix = "multi_" if multi else ""
    prefix += "under_" if under_sample else ""

    indices_used = np.asarray(range(len(labels))).reshape(-1, 1)

    if under_sample:
        print("Under sampling ...")
        from imblearn.under_sampling import RandomUnderSampler
        indices_used, _ = RandomUnderSampler(random_state=42).fit_resample(indices_used, labels)

    indices_used = np.sort(indices_used.reshape(-1).astype(int))

    if 0 < limit < len(indices_used):
        if under_sample:
            indices_used = np.sort(np.random.choice(indices_used, limit * 2))
        else:
            indices_used = indices_used[:limit]

        if sum(labels) <= 0:
            raise ValueError("Not enough positive labels. Increase limit!")

    features = np.array(features)
    labels = np.array(labels)
    weights = np.array(weights)
    extra_features = np.array(extra_features)

    print("Saving labels ...")
    joblib.dump(labels[indices_used], join(output_path, prefix + 'labels.npz'), compress=True)

    print("Saving sample weights ...")
    joblib.dump(weights[indices_used], join(output_path, prefix + 'sample_weights.npz'), compress=True)

    if extra:
        print("Saving extra features ...")
        joblib.dump(extra_features[indices_used], join(output_path, prefix + 'extra_features.npz'), compress=True)

    if multi:
        print("Saving multi-features ...")
        joblib.dump(features, join(output_path, prefix + 'dataset_features.npz'), compress=True)
        print("Saving low scaler ...")
        pickle.dump(StandardScaler().fit(features[:, :, 0]), open(join(output_path, prefix + 'scaler_low.pkl'), 'wb'))
        print("Saving mid scaler ...")
        pickle.dump(StandardScaler().fit(features[:, :, 1]), open(join(output_path, prefix + 'scaler_mid.pkl'), 'wb'))
        print("Saving high scaler ...")
        pickle.dump(StandardScaler().fit(features[:, :, 2]), open(join(output_path, prefix + 'scaler_high.pkl'), 'wb'))
    else:
        print("Saving features ...")
        joblib.dump(features[indices_used], join(output_path, prefix + 'dataset_features.npz'), compress=True)
        print("Saving scaler ...")
        pickle.dump(StandardScaler().fit(features), open(join(output_path, prefix + 'scaler.pkl'), 'wb'))


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

    training_data_collection(wavs_path=args.wav, timings_path=args.timing, output_path=args.output,
                             multi_int=args.multi, extra_int=args.extra, under_sample_int=args.under_sample,
                             limit=args.limit)
