import os
import re

import numpy as np

from stepcovnet.configuration.parameters import NUM_MULTI_CHANNELS, NUM_FREQ_BANDS, NUM_TIME_BANDS


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


def get_scalers(features, multi):
    """
    Gather scalers along the frequency axis
    :param features: nparray - audio feature frames reshaped into time and frequency axises
    :param multi: bool - True if using multiple channels
    :return scalers: list - mean and std for each frequency band for each channel
    """
    # TODO: Add better check for non formatted features
    if set(features.shape[1:]).intersection({NUM_FREQ_BANDS * NUM_TIME_BANDS * NUM_MULTI_CHANNELS, NUM_FREQ_BANDS * NUM_TIME_BANDS}):
        raise ValueError('Need to reshape features before getting scalers')

    scalers = []
    if multi:
        import multiprocessing
        import psutil
        with multiprocessing.Pool(psutil.cpu_count(logical=False)) as pool:
            for result in pool.imap(get_features_mean_std, (features[:, i] for i in range(NUM_FREQ_BANDS))):
                scalers.append(result)
        return np.array(scalers).T
    else:
        return np.array(get_features_mean_std(np.transpose(features, (0, 2, 1))))


def pre_process(features, multi, labels=None, extra_features=None, scalers=None):
    features_copy = np.copy(features)
    if multi:
        if scalers is not None:
            scalers_copy = np.transpose(np.copy(scalers), (2, 1, 0))
            for i, scaler in enumerate(scalers_copy):
                features_copy[:, i] = (features_copy[:, i] - scaler[0]) / scaler[1]
    else:
        if scalers is not None:
            for i, scaler in enumerate(scalers.T):
                features_copy[:, i] = (features_copy[:, i] - scaler[0]) / scaler[1]
        features_copy = np.expand_dims(np.squeeze(features_copy), axis=1)

    if extra_features is not None:
        features_copy = [features_copy, extra_features]

    if labels is not None:
        labels = labels.reshape(-1, 1)
        return features_copy, labels
    else:
        return features_copy
