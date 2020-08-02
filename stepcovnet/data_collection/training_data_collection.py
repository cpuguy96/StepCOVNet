import json
import multiprocessing
import os
import pickle
import time
from datetime import datetime
from functools import partial
from os.path import join

import psutil

from stepcovnet.common.parameters import CONFIG
from stepcovnet.common.parameters import VGGISH_CONFIG
from stepcovnet.common.utils import get_channel_scalers
from stepcovnet.common.utils import get_filename
from stepcovnet.common.utils import get_filenames_from_folder
from stepcovnet.data_collection.sample_collection_helper import feature_onset_phrase_label_sample_weights
from stepcovnet.data_collection.sample_collection_helper import get_features_and_labels
from stepcovnet.dataset.ModelDatasetTypes import ModelDatasetTypes


def build_all_metadata(**kwargs):
    kwargs["creation_time"] = datetime.utcnow().strftime("%b %d %Y %H:%M:%S UTC")
    return kwargs


def update_all_metadata(all_metadata, metadata):
    for key, value in metadata.items():
        if isinstance(value, list):
            if key not in all_metadata:
                all_metadata[key] = [value]
            else:
                all_metadata[key].append(value)
        else:
            all_metadata[key] = value
    return all_metadata


def collect_features(wav_path, timing_path, multi, config, file_name):
    # from the annotation to get feature, frame start and frame end of each line, frames_onset
    try:
        print('Feature collecting: %s' % file_name)
        # New version. Currently has memory constrain issues.
        # With enough memory, this method should perform much faster.
        log_mel, onsets, arrows, label_encoded_arrows, binary_encoded_arrows = get_features_and_labels(wav_path,
                                                                                                       timing_path,
                                                                                                       file_name, multi,
                                                                                                       config)
        # Old version
        # log_mel, onsets, arrows, encoded_arrows =
        # get_features_and_labels_madmom(wav_path, timing_path, file_name, multi, config)
        feature, label_dict, sample_weights_dict, arrows_dict, label_encoded_arrows_dict, binary_encoded_arrows_dict = \
            feature_onset_phrase_label_sample_weights(onsets, log_mel, arrows, label_encoded_arrows,
                                                      binary_encoded_arrows, config["NUM_ARROW_TYPES"])

        # type casting features to float16 to save disk space.
        return [file_name, feature.astype("float16"), label_dict, sample_weights_dict, arrows_dict,
                label_encoded_arrows_dict, binary_encoded_arrows_dict]
    except Exception as ex:
        print("Error collecting features for %s: %r" % (file_name, ex))
        return None


def collect_data(wavs_path, timings_path, output_path, name_prefix, config, training_dataset, dataset_type, multi=False,
                 limit=-1, cores=1):
    func = partial(collect_features, wavs_path, timings_path, multi, config)
    file_names = [get_filename(file_name, with_ext=False) for file_name in get_filenames_from_folder(timings_path)]

    scalers = None
    all_metadata = build_all_metadata(dataset_name=name_prefix, dataset_type=dataset_type.name, config=config)

    with training_dataset as dataset:
        with multiprocessing.Pool(cores) as pool:
            song_count = 0
            for i, result in enumerate(pool.imap(func, file_names)):
                if result is None:
                    continue
                file_name, features, labels, weights, arrows, label_encoded_arrows, binary_encoded_arrows = result
                print("[%d/%d] Dumping to dataset: %s" % (i + 1, len(file_names), file_name))
                dataset.dump(features=features, labels=labels, sample_weights=weights, arrows=arrows,
                             label_encoded_arrows=label_encoded_arrows, binary_encoded_arrows=binary_encoded_arrows,
                             file_names=file_name)
                # not using joblib parallel since we are already using multiprocessing
                print("[%d/%d] Creating scalers: %s" % (i + 1, len(file_names), file_name))
                scalers = get_channel_scalers(features, existing_scalers=scalers, n_jobs=1)
                all_metadata = update_all_metadata(all_metadata, {"file_name": file_name})
                # Save scalers after every 10 runs
                if i % 10 == 0:
                    print("Saving scalers")
                    pickle.dump(scalers, open(join(output_path, name_prefix + '_scaler.pkl'), 'wb'))
                if limit > 0:
                    song_count += 1
                    print("[%d/%d] Features collected: %s " % (dataset.num_samples, limit, file_name))
                    if dataset.num_samples >= limit:
                        print("Limit reached after %d songs. Breaking..." % song_count)
                        break
    print("Saving scalers")
    pickle.dump(scalers, open(join(output_path, name_prefix + '_scaler.pkl'), 'wb'))
    print("Saving metadata")
    with open(join(output_path, 'metadata.json'), 'w') as json_file:
        json_file.write(json.dumps(all_metadata))


def training_data_collection(wavs_path, timings_path, output_path, multi_int, type_int, limit, cores, name,
                             distributed_int):
    if not os.path.isdir(wavs_path):
        raise NotADirectoryError('Audio path %s not found' % wavs_path)

    if not os.path.isdir(timings_path):
        raise NotADirectoryError('Annotation path %s not found' % timings_path)

    if limit == 0:
        raise ValueError('Limit cannot be 0!')

    if name is not None and not name:
        raise ValueError('Model name cannot be empty')

    if cores > os.cpu_count() or cores == 0:
        raise ValueError(
            'Number of cores selected must not be 0 and must be less than the number cpu cores (%d)' % os.cpu_count())

    multi = True if multi_int == 1 else False
    config = VGGISH_CONFIG if type_int == 1 else CONFIG
    limit = max(-1, limit)  # defaulting negative inputs to -1
    cores = psutil.cpu_count(logical=False) if cores < 0 else cores
    distributed = True if distributed_int == 1 else False

    prefix = "multi_%d_channel_" % config["NUM_MULTI_CHANNELS"] if multi else ""
    name_prefix = name if name is not None else prefix + "stepcovnet"
    name_postfix = "" if distributed is False else "_distributed"
    name_postfix += "_dataset"

    output_path = os.path.join(output_path, name_prefix + name_postfix)
    os.makedirs(output_path, exist_ok=True)
    dataset_type = ModelDatasetTypes.DISTRIBUTED_DATASET if distributed else ModelDatasetTypes.SINGULAR_DATASET
    training_dataset = dataset_type.value(os.path.join(output_path, name_prefix + name_postfix), overwrite=True)

    start_time = time.time()
    collect_data(wavs_path=wavs_path, timings_path=timings_path, output_path=output_path, name_prefix=name_prefix,
                 config=config, multi=multi, limit=limit, cores=cores, training_dataset=training_dataset,
                 dataset_type=dataset_type)
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
    parser.add_argument("--distributed",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="Whether to create a single dataset or a distributed dataset: 0 - single, 1 - distributed")
    args = parser.parse_args()

    training_data_collection(wavs_path=args.wav, timings_path=args.timing, output_path=args.output,
                             multi_int=args.multi, type_int=args.type, limit=args.limit, cores=args.cores,
                             name=args.name, distributed_int=args.distributed)
