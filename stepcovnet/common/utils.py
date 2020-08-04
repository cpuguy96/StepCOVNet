import multiprocessing
import os
import re
import time

import numpy as np
import psutil
from joblib import Parallel
from joblib import delayed
from nltk.util import ngrams
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from stepcovnet.common.parameters import NUM_FREQ_BANDS
from stepcovnet.common.parameters import NUM_MULTI_CHANNELS
from stepcovnet.common.parameters import NUM_TIME_BANDS


def timed_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        return_value = func(*args, **kwargs)
        end_time = time.time()
        print("\nElapsed time was %g seconds for %s" % ((end_time - start_time), func.__name__))
        return return_value

    return wrapper


def timed(func, *args, **kwargs):
    start_time = time.time()
    return_value = func(*args, **kwargs)
    end_time = time.time()
    print("\nElapsed time was %g seconds for %s" % ((end_time - start_time), func.__name__))
    return return_value


def get_filenames_from_folder(mypath):
    return [file for file in os.listdir(mypath)
            if os.path.isfile(os.path.join(mypath, file)) and
            os.path.splitext(file)[0] not in {".DS_Store", "_DS_Store"}]


def get_filename(file_path, with_ext=True):
    if with_ext:
        return os.path.basename(file_path)
    else:
        return os.path.splitext(os.path.basename(file_path))[0]


def standardize_filename(filename):
    return re.sub("[^a-z0-9-_]", "", filename.lower())


def write_file(output_path, output_data, header=""):
    with open(output_path, "w") as file:
        file.write(header + output_data)


def feature_reshape_down(features, order='C'):
    if len(features.shape) != 4:
        raise ValueError('Number of dims for features is %d (should be 4)')

    return features.reshape(features.shape[0], features.shape[1] * features.shape[2], features.shape[3], order=order)


def feature_reshape_up(feature, num_freq_bands, num_time_bands, num_channels, order='F'):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param order:
    :param num_channels:
    :param num_time_bands:
    :param num_freq_bands:
    :param feature:
    :return:
    """
    return feature.reshape((len(feature), num_time_bands, num_freq_bands, num_channels), order=order)


def get_channel_scalers(features, existing_scalers=None, n_jobs=1):
    if len(features.shape) not in {3, 4}:
        raise ValueError('Number of dims for features is %d (should be 3 or 4)' % len(features.shape))

    if len(features.shape) == 4:
        features = feature_reshape_down(features=features)

    num_channels = features.shape[-1]

    if existing_scalers is None:
        channel_scalers = [StandardScaler() for _ in range(num_channels)]
    else:
        channel_scalers = existing_scalers
    channel_scalers = Parallel(backend="loky", n_jobs=n_jobs)(
        delayed(scaler.partial_fit)(features[:, :, i]) for i, scaler in
        enumerate(channel_scalers))

    return channel_scalers


def get_features_mean_std(features):
    return [np.mean(np.mean(features, axis=1), axis=0, dtype="float32"),
            np.mean(np.std(features, axis=1), axis=0, dtype="float32")]


def sklearn_scaler_to_numpy(scalers, multi):
    np_scaler = []
    if multi:
        for scaler in scalers:
            np_scaler.append([[sca.mean_[0], np.sqrt(sca.var_[0])] for sca in scaler])
    else:
        np_scaler = [[scaler.mean_[0], np.sqrt(scaler.var_[0])] for scaler in scalers]

    return np.array(np_scaler)


def parital_fit_feature_slice(scaler, feat_slice):
    return scaler.partial_fit(np.mean(feat_slice, axis=1).reshape(-1, 1))  # need to take mean slice has 15 time bands


def get_sklearn_scalers(features, multi, config, existing_scalers=None, parallel=True):
    # if set(features.shape[1:]).intersection(
    #        {NUM_FREQ_BANDS * NUM_TIME_BANDS * NUM_MULTI_CHANNELS, NUM_FREQ_BANDS * NUM_TIME_BANDS}):
    #    raise ValueError('Need to reshape features before getting scalers')

    scalers = []
    n_jobs = psutil.cpu_count(logical=False) if parallel else 1

    if multi:
        for channel in range(NUM_MULTI_CHANNELS):
            if existing_scalers is not None:
                channel_scalers = existing_scalers[channel]
            else:
                channel_scalers = [StandardScaler() for _ in range(config["NUM_FREQ_BANDS"])]
            feat_slice_gen = (features[:, i, :, channel] for i in range(config["NUM_FREQ_BANDS"]))
            channel_scalers = Parallel(backend="loky", n_jobs=n_jobs)(
                delayed(parital_fit_feature_slice)(sca, feat_slice) for sca, feat_slice in
                zip(channel_scalers, feat_slice_gen))
            scalers.append(channel_scalers)
        return scalers
    else:
        if existing_scalers is not None:
            scalers = existing_scalers
        else:
            scalers = [StandardScaler() for _ in range(config["NUM_FREQ_BANDS"])]
        feat_slice_gen = (features[:, :, i] for i in range(config["NUM_FREQ_BANDS"]))
        scalers = Parallel(backend="loky", n_jobs=n_jobs)(
            delayed(parital_fit_feature_slice)(sca, feat_slice) for sca, feat_slice in zip(scalers, feat_slice_gen))
        return scalers


def get_scalers(features, multi):
    """
    Gather scalers along the frequency axis
    :param features: nparray - audio feature frames reshaped into time and frequency axises
    :param multi: bool - True if using multiple channels
    :return scalers: list - mean and std for each frequency band for each channel
    """
    # TODO: Add better check for non formatted features
    if set(features.shape[1:]).intersection(
            {NUM_FREQ_BANDS * NUM_TIME_BANDS * NUM_MULTI_CHANNELS, NUM_FREQ_BANDS * NUM_TIME_BANDS}):
        raise ValueError('Need to reshape features before getting scalers')

    scalers = []
    if multi:
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            for result in pool.imap(get_features_mean_std, (features[:, i] for i in range(NUM_FREQ_BANDS))):
                scalers.append(result)
        return np.array(scalers).T
    else:
        return np.array(get_features_mean_std(np.transpose(features, (0, 2, 1))))


def apply_scalers(features, scalers):
    if scalers is None:
        return features
    if features.shape not in {3, 4}:
        raise ValueError('Features dimensions must be 3 or 4 (received dimension %d)' % len(features.shape))
    original_features_shape = features.shape
    if len(features.shape) == 4:
        features = feature_reshape_down(features=features)
    if type(scalers) is not list:
        scalers = [scalers]
    if len(scalers) != features.shape[-1]:
        raise ValueError('Number of scalers (%d) does not equal number of feature channels (%d)' % (
            len(scalers), features.shape[-1]))
    for i, scaler in enumerate(scalers):
        features[:, :, i] = scaler.transform(features[:, :, i])

    if len(original_features_shape) == 4:
        return feature_reshape_up(features, original_features_shape[1], original_features_shape[2],
                                  original_features_shape[3], order='C')
    return features


def pre_process(features, labels=None, multi=False, scalers=None):
    features_copy = np.copy(features)
    if multi:
        if scalers is not None:
            for i, scaler_channel in enumerate(scalers):
                for j, scaler in enumerate(scaler_channel):
                    features_copy[:, j, :, i] = scaler.transform(features_copy[:, j, :, i])
    else:
        if scalers is not None:
            for j, scaler in enumerate(scalers):
                features_copy[:, j] = scaler.transform(features_copy[:, j])
        features_copy = np.expand_dims(np.squeeze(features_copy), axis=1)

    if labels is not None:
        return features_copy, labels
    else:
        return features_copy


def get_all_note_combs():
    all_note_combs = []

    for first_digit in range(0, 4):
        for second_digit in range(0, 4):
            for third_digit in range(0, 4):
                for fourth_digit in range(0, 4):
                    all_note_combs.append(
                        str(first_digit) + str(second_digit) + str(third_digit) + str(fourth_digit))
    # Adding '0000' to possible note combinations. This will allow the arrow prediction model to predict an empty note.
    # all_note_combs = all_note_combs[1:]

    return all_note_combs


def get_arrow_one_hot_encoder():
    return OneHotEncoder(categories='auto', sparse=False).fit(np.asarray(get_all_note_combs()).reshape(-1, 1))


def get_arrow_label_encoder():
    return LabelEncoder().fit(np.asarray(get_all_note_combs()).reshape(-1, 1))


def get_ngram(data, lookback, padding_value=0):
    padding = np.full((lookback,) + data.shape[1:], fill_value=padding_value)
    data_w_padding = np.append(padding, data, axis=0)
    return np.asarray(list(ngrams(data_w_padding, lookback)))
