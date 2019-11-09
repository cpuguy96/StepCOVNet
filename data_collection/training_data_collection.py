from common.audio_preprocessing import getMFCCBands2DMadmom, get_madmom_librosa_features
from common.parameters import *
from data_collection.sample_collection_helper import feature_onset_phrase_label_sample_weights

from os.path import join
from sklearn.preprocessing import StandardScaler

import os
import joblib
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


def dump_feature_label_sample_weights_onset_phrase(audio_path,
                                                   annotation_path,
                                                   path_output,
                                                   multi,
                                                   extra,
                                                   under_sample,
                                                   is_limited,
                                                   limit):
    """
    dump feature, label, sample weights for each phrase with bock annotation format
    :param audio_path:
    :param annotation_path:
    :param path_output:
    :return:
    """

    features_low = []
    features_mid = []
    features_high = []

    features = []

    extra_features = []
    labels = []
    weights = []

    sample_count = 0

    if multi:
        channel = 3
    else:
        channel = 1

    if is_limited and under_sample:
        limit //= 2

    for i, fn in enumerate(getRecordings(annotation_path)):
        if is_limited and under_sample and sample_count >= limit:
            print("limit reached after %d songs. breaking..." % i)
            break
        elif is_limited and not under_sample and sample_count >= limit:
            print("limit reached after %d songs. breaking..." % i)
            break

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
                extra_feature = None

        except Exception:
            print("Error collecting features for", fn)
            continue

        if multi:
            features_low.append(feature[:, :, 0].astype("float16"))
            features_mid.append(feature[:, :, 1].astype("float16"))
            features_high.append(feature[:, :, 2].astype("float16"))
        else:
            features.append(feature)

        labels.append(label)
        weights.append(sample_weights)

        if extra:
            extra_features.append(extra_feature)

        if is_limited and under_sample:
            sample_count += label.sum()
        elif is_limited:
            sample_count += len(label)

    labels = np.array(np.concatenate(labels, axis=0)).astype("int8")
    weights = np.array(np.concatenate(weights, axis=0)).astype("float16")

    if extra:
        extra_features = np.array(np.concatenate(extra_features, axis=0)).astype("int8")

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
    np.savez_compressed(join(path_output, prefix + 'labels'), labels=labels[indices_used])

    print("Saving sample weights ...")
    np.savez_compressed(join(path_output, prefix + 'sample_weights'), sample_weights=weights[indices_used])

    if extra:
        print("Saving extra features ...")
        np.savez_compressed(join(path_output, prefix + 'extra_features'), extra_features=extra_features[indices_used])

    if multi:
        features_low = np.array(np.concatenate(features_low, axis=0)[indices_used].astype("float16"))
        features_mid = np.array(np.concatenate(features_mid, axis=0)[indices_used].astype("float16"))
        features_high = np.array(np.concatenate(features_high, axis=0)[indices_used].astype("float16"))
        stacked_feats = np.stack([features_low, features_mid, features_high], axis=-1).astype("float16")

        print("Saving multi-features ...")
        np.savez_compressed(join(path_output, prefix + 'dataset_features'), features=stacked_feats)
        print("Saving low scaler ...")
        joblib.dump(StandardScaler().fit(features_low), join(path_output, prefix + 'scaler_low.pkl'))
        print("Saving mid scaler ...")
        joblib.dump(StandardScaler().fit(features_mid), join(path_output, prefix + 'scaler_mid.pkl'))
        print("Saving high scaler ...")
        joblib.dump(StandardScaler().fit(features_mid), join(path_output, prefix + 'scaler_high.pkl'))
    else:
        features = np.array(np.concatenate(features, axis=0)).astype("float16")
        print("Saving features ...")
        np.savez_compressed(join(path_output, prefix + 'dataset_features'), features=features[indices_used])
        print("Saving scaler ...")
        joblib.dump(StandardScaler().fit(features), join(path_output, prefix + 'scaler.pkl'))


def main():
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

    if not os.path.isdir(args.audio):
        raise NotADirectoryError('Audio path %s not found' % args.audio)

    if not os.path.isdir(args.annotation):
        raise NotADirectoryError('Annotation path %s not found' % args.annotation)

    if not os.path.isdir(args.output):
        print('Output path not found. Creating directory...')
        os.makedirs(args.output, exist_ok=True)

    if args.multi == 1:
        multi = True
    else:
        multi = False

    if args.extra == 1:
        extra = True
    else:
        extra = False

    if args.under_sample == 1:
        under_sample = True
    else:
        under_sample = False

    if args.limit == 0:
        raise ValueError('Limit cannot be 0!')

    if args.limit < 0:
        is_limited = False
    else:
        is_limited = True

    dump_feature_label_sample_weights_onset_phrase(audio_path=args.audio,
                                                   annotation_path=args.annotation,
                                                   path_output=args.output,
                                                   multi=multi,
                                                   extra=extra,
                                                   under_sample=under_sample,
                                                   is_limited=is_limited,
                                                   limit=args.limit)


if __name__ == '__main__':
    main()
