import multiprocessing
import os
import pickle
import time
from functools import partial
from os.path import join

import psutil

from stepcovnet.common.audio_preprocessing import get_madmom_librosa_features
from stepcovnet.common.model_dataset import ModelDataset
from stepcovnet.common.parameters import HOPSIZE_T
from stepcovnet.common.parameters import SAMPLE_RATE
from stepcovnet.common.utils import get_filename
from stepcovnet.common.utils import get_filenames_from_folder
from stepcovnet.common.utils import get_sklearn_scalers
from stepcovnet.data_collection.sample_collection_helper import dump_feature_onset_helper
from stepcovnet.data_collection.sample_collection_helper import feature_onset_phrase_label_sample_weights


def collect_features(wav_path, timing_path, multi, extra, file_name):
    # from the annotation to get feature, frame start and frame end of each line, frames_onset
    try:
        print('Feature collecting: %s' % file_name)
        log_mel, frames_onset, arrows = \
            dump_feature_onset_helper(wav_path, timing_path, file_name, multi)

        # simple sample weighting
        feature, label_dict, sample_weights_dict, arrows_dict = \
            feature_onset_phrase_label_sample_weights(frames_onset, log_mel, arrows)

        if extra:
            # beat frames predicted by madmom DBNBeatTrackingProcess and librosa.onset.onset_decect
            extra_feature = get_madmom_librosa_features(join(wav_path, file_name + '.wav'),
                                                        SAMPLE_RATE,
                                                        HOPSIZE_T,
                                                        len(feature))
        else:
            extra_feature = None
        # type casting features to float16 to save disk space
        return [feature.astype("float16"), label_dict, sample_weights_dict, extra_feature, arrows_dict]
    except Exception as ex:
        print("Error collecting features for %s: %r" % (file_name, ex))
        return None


def collect_data(wavs_path, timings_path, multi, extra, limit, output_path, prefix):
    func = partial(collect_features, wavs_path, timings_path, multi, extra)
    file_names = [get_filename(file_name, with_ext=False) for file_name in get_filenames_from_folder(timings_path)]

    scalers = None

    with ModelDataset(os.path.join(output_path, prefix + "stepcovnet_dataset.hdf5"), overwrite=True) as dataset:
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            song_count = 0
            for result in pool.imap(func, file_names):
                if result is None:
                    continue
                features, labels, weights, extra_features, arrows = result
                dataset.dump(features, labels, weights, extra_features, arrows)
                # not using joblib parallel since we are already using multiprocessing
                scalers = get_sklearn_scalers(features, multi, scalers, parallel=False)
                if limit > 0:
                    song_count += 1
                    if dataset.num_valid_samples >= limit:
                        print("Limit reached after %d songs. Breaking..." % song_count)
                        break
    if multi:
        # TODO: Change to allow any number of channels
        print("Saving low scaler ...")
        pickle.dump(scalers[0], open(join(output_path, prefix + 'scaler_low.pkl'), 'wb'))
        print("Saving mid scaler ...")
        pickle.dump(scalers[1], open(join(output_path, prefix + 'scaler_mid.pkl'), 'wb'))
        print("Saving high scaler ...")
        pickle.dump(scalers[2], open(join(output_path, prefix + 'scaler_high.pkl'), 'wb'))
    else:
        print("Saving scaler ...")
        pickle.dump(scalers, open(join(output_path, prefix + 'scaler.pkl'), 'wb'))


def training_data_collection(wavs_path, timings_path, output_path, multi_int, extra_int, limit):
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
    limit = max(-1, limit)  # defaulting negative inputs to -1

    prefix = "multi_" if multi else ""

    start_time = time.time()
    collect_data(wavs_path, timings_path, multi, extra, limit, output_path, prefix)
    end_time = time.time()

    print("\nElapsed time was %g seconds" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Collect audio and timings data to create training dataset")
    parser.add_argument("-w", "--wav",
                        type=str,
                        help="Input wavs path")
    parser.add_argument("-t", "--timing",
                        type=str,
                        help="Input timings path")
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output collected data path")
    parser.add_argument("--multi",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Whether multiple STFT window time-lengths are captured: 0 - single, 1 - multi")
    parser.add_argument("--extra",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Whether to gather extra data from madmom and librosa: 0 - not collected, 1 - collected")
    parser.add_argument("--limit",
                        type=int,
                        default=-1,
                        help="Maximum number of frames allowed to be collected: -1 unlimited, > 0 frame limit")
    args = parser.parse_args()

    training_data_collection(wavs_path=args.wav, timings_path=args.timing, output_path=args.output,
                             multi_int=args.multi, extra_int=args.extra,
                             limit=args.limit)
