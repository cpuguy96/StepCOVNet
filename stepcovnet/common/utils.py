import multiprocessing
import os
import re

import numpy as np
import psutil
from joblib import Parallel
from joblib import delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from stepcovnet.common.parameters import NUM_FREQ_BANDS
from stepcovnet.common.parameters import NUM_MULTI_CHANNELS
from stepcovnet.common.parameters import NUM_TIME_BANDS


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


def feature_reshape(feature, multi=False):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """
    if multi:
        return feature.reshape((len(feature), NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS), order='F')
    else:
        return feature.reshape((len(feature), NUM_FREQ_BANDS, NUM_TIME_BANDS), order='F')


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


def get_sklearn_scalers(features, multi, existing_scalers=None, parallel=True):
    from sklearn.preprocessing import StandardScaler
    if set(features.shape[1:]).intersection(
            {NUM_FREQ_BANDS * NUM_TIME_BANDS * NUM_MULTI_CHANNELS, NUM_FREQ_BANDS * NUM_TIME_BANDS}):
        raise ValueError('Need to reshape features before getting scalers')

    scalers = []
    n_jobs = psutil.cpu_count(logical=False) if parallel else 1

    if multi:
        for channel in range(NUM_MULTI_CHANNELS):
            if existing_scalers is not None:
                channel_scalers = existing_scalers[channel]
            else:
                channel_scalers = [StandardScaler() for _ in range(NUM_FREQ_BANDS)]
            feat_slice_gen = (features[:, i, :, channel] for i in range(NUM_FREQ_BANDS))
            channel_scalers = Parallel(backend="loky", n_jobs=n_jobs)(
                delayed(parital_fit_feature_slice)(sca, feat_slice) for sca, feat_slice in
                zip(channel_scalers, feat_slice_gen))
            scalers.append(channel_scalers)
        return scalers
    else:
        if existing_scalers is not None:
            scalers = existing_scalers
        else:
            scalers = [StandardScaler() for _ in range(NUM_FREQ_BANDS)]
        feat_slice_gen = (features[:, i] for i in range(NUM_FREQ_BANDS))
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


def pre_process(features, labels=None, extra_features=None, multi=False, scalers=None):
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

    if extra_features is not None:
        features_copy = {"log_mel_input": features_copy, "extra_input": extra_features}

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
