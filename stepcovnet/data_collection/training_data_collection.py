import multiprocessing
import os
import pickle
import time
from functools import partial
from os.path import join

import psutil

from stepcovnet.common.audio_preprocessing import get_madmom_librosa_features
from stepcovnet.common.model_dataset import ModelDataset
from stepcovnet.common.parameters import CONFIG
from stepcovnet.common.parameters import HOPSIZE_T
from stepcovnet.common.parameters import SAMPLE_RATE
from stepcovnet.common.parameters import VGGISH_CONFIG
from stepcovnet.common.utils import get_channel_scalers
from stepcovnet.common.utils import get_filename
from stepcovnet.common.utils import get_filenames_from_folder
from stepcovnet.data_collection.sample_collection_helper import feature_onset_phrase_label_sample_weights
from stepcovnet.data_collection.sample_collection_helper import get_features_and_labels


def collect_features(wav_path, timing_path, multi, extra, config, file_name):
    # from the annotation to get feature, frame start and frame end of each line, frames_onset
    try:
        print('Feature collecting: %s' % file_name)
        # New version. Currently has memory constrain issues.
        # With enough memory, this method should perform much faster.
        log_mel, onsets, arrows = get_features_and_labels(wav_path, timing_path, file_name, multi, config)
        # Old version
        # log_mel, onsets, arrows = get_features_and_labels_madmom(wav_path, timing_path, file_name, multi, config)

        # simple sample weighting
        feature, label_dict, sample_weights_dict, arrows_dict = \
            feature_onset_phrase_label_sample_weights(onsets, log_mel, arrows)

        if extra:
            # beat frames predicted by madmom DBNBeatTrackingProcess and librosa.onset.onset_decect
            extra_feature = get_madmom_librosa_features(join(wav_path, file_name + '.wav'),
                                                        SAMPLE_RATE,
                                                        HOPSIZE_T,
                                                        len(feature))
        else:
            extra_feature = None
        # type casting features to float16 to save disk space.
        return [feature.astype("float16"), label_dict, sample_weights_dict, extra_feature, arrows_dict]
    except Exception as ex:
        print("Error collecting features for %s: %r" % (file_name, ex))
        return None


def collect_data(wavs_path, timings_path, output_path, name_prefix, config, multi=False, extra=False, limit=-1,
                 cores=1):
    func = partial(collect_features, wavs_path, timings_path, multi, extra, config)
    file_names = [get_filename(file_name, with_ext=False) for file_name in get_filenames_from_folder(timings_path)]

    scalers = None

    with ModelDataset(os.path.join(output_path, name_prefix + "_dataset.hdf5"), overwrite=True) as dataset:
        with multiprocessing.Pool(cores) as pool:
            song_count = 0
            for i, result in enumerate(pool.imap(func, file_names)):
                if result is None:
                    continue
                features, labels, weights, extra_features, arrows = result
                dataset.dump(features, labels, weights, extra_features, arrows)
                # not using joblib parallel since we are already using multiprocessing
                scalers = get_channel_scalers(features, existing_scalers=scalers, n_jobs=1)
                # Save scalers after every run
                pickle.dump(scalers, open(join(output_path, name_prefix + '_scaler.pkl'), 'wb'))
                if limit > 0:
                    song_count += 1
                    if dataset.num_valid_samples >= limit:
                        print("Limit reached after %d songs. Breaking..." % song_count)
                        break


def training_data_collection(wavs_path, timings_path, output_path, multi_int, extra_int, type_int, limit, cores, name):
    if not os.path.isdir(wavs_path):
        raise NotADirectoryError('Audio path %s not found' % wavs_path)

    if not os.path.isdir(timings_path):
        raise NotADirectoryError('Annotation path %s not found' % timings_path)

    if not os.path.isdir(output_path):
        print('Output path not found. Creating directory...')
        os.makedirs(output_path, exist_ok=True)

    if limit == 0:
        raise ValueError('Limit cannot be 0!')

    if name is not None and not name:
        raise ValueError('Model name cannot be empty')

    if cores > os.cpu_count():
        raise ValueError('Number of cores selected cannot be greater than the number cpu cores (%d)' % os.cpu_count())

    multi = True if multi_int == 1 else False
    extra = True if extra_int == 1 else False
    config = VGGISH_CONFIG if type_int == 1 else CONFIG
    limit = max(-1, limit)  # defaulting negative inputs to -1
    cores = psutil.cpu_count(logical=False) if cores < 0 else cores

    prefix = "multi_%d_channel_" % config["NUM_MULTI_CHANNELS"] if multi else ""

    name_prefix = name if name is not None else prefix + "stepcovnet"

    start_time = time.time()
    collect_data(wavs_path=wavs_path, timings_path=timings_path, output_path=output_path, name_prefix=name_prefix,
                 config=config, multi=multi, extra=extra, limit=limit, cores=cores)
    end_time = time.time()

    print("\nElapsed time was %g seconds" % (end_time - start_time))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Collect audio and timings data to create training dataset")
    parser.add_argument("-w", "--wav",
                        type=str,
                        required=True,
                        help="Input wavs path")
    parser.add_argument("-t", "--timing",
                        type=str,
                        required=True,
                        help="Input timings path")
    parser.add_argument("-o", "--output",
                        type=str,
                        required=True,
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
    parser.add_argument("--type",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Whether to preprocess audio data for VGGish model or custom model: 0 - custom model, "
                             "1 - VGGish")
    parser.add_argument("--limit",
                        type=int,
                        default=-1,
                        help="Maximum number of frames allowed to be collected: -1 unlimited, > 0 frame limit")
    parser.add_argument("--cores",
                        type=int,
                        default=1,
                        help="Number of processor cores to use for parallel processing: -1 max number of physical cores")
    parser.add_argument("--name",
                        type=str,
                        default=None,
                        help="Name to give model dataset")
    args = parser.parse_args()

    training_data_collection(wavs_path=args.wav, timings_path=args.timing, output_path=args.output,
                             multi_int=args.multi, extra_int=args.extra, type_int=args.type,
                             limit=args.limit, cores=args.cores, name=args.name)
