import os
import re
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import librosa
import numpy as np
from nltk.util import ngrams
from sklearn.preprocessing import StandardScaler


def get_filenames_from_folder(mypath: str) -> Sequence[str]:
    return [file for file in os.listdir(mypath)
            if os.path.isfile(os.path.join(mypath, file)) and
            os.path.splitext(file)[0] not in {".DS_Store", "_DS_Store"}]


def get_filename(file_path: str, with_ext: bool = True) -> str:
    if with_ext:
        return os.path.basename(file_path)
    else:
        return os.path.splitext(os.path.basename(file_path))[0]


def standardize_filename(filename: str) -> str:
    return re.sub(' ', '_', re.sub('[^a-z0-9-_ ]', '', filename.lower()))


def write_file(output_path: str, output_data, header: str = ""):
    with open(output_path, "w") as file:
        file.write(header + output_data)


def get_bpm(wav_file_path: str) -> float:
    y, sr = librosa.load(wav_file_path)
    return librosa.beat.beat_track(y=y, sr=sr)[0]


def feature_reshape_down(features: np.ndarray, order: str = 'C') -> np.ndarray:
    if len(features.shape) != 4:
        raise ValueError('Number of dims for features is %d (should be 4)')

    return features.reshape((features.shape[0], features.shape[1] * features.shape[2], features.shape[3]),
                            order=order)


def feature_reshape_up(feature: np.ndarray, num_freq_bands: int, num_time_bands: int, num_channels: int,
                       order: str = 'F'):
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


def get_channel_scalers(features: np.ndarray, existing_scalers: Sequence[StandardScaler] = None):
    if len(features.shape) not in {3, 4}:
        raise ValueError('Number of dims for features is %d (should be 3 or 4)' % len(features.shape))

    if len(features.shape) == 4:
        features = feature_reshape_down(features=features)

    num_channels = features.shape[-1]
    channel_scalers = existing_scalers if existing_scalers is not None \
        else [StandardScaler() for _ in range(num_channels)]
    channel_scalers = [scaler.partial_fit(features[:, :, i]) for i, scaler in enumerate(channel_scalers)]

    return channel_scalers


def apply_timeseries_scalers(features: np.ndarray, scalers: Union[StandardScaler, List[StandardScaler]]):
    if scalers is None:
        return features
    if len(features.shape) not in {4, 5}:
        raise ValueError('Features dimensions must be 4 or 5 (received dimension %d)' % len(features.shape))
    if len(features.shape) == 4:
        for time_slice in range(features.shape[0]):
            features[time_slice] = apply_scalers(features=features[time_slice], scalers=scalers)
    else:
        for time_slice in range(features.shape[1]):
            features[:, time_slice] = apply_scalers(features=features[:, time_slice], scalers=scalers)
    return features


def apply_scalers(features: np.ndarray, scalers: Union[StandardScaler, List[StandardScaler]]) -> np.ndarray:
    if scalers is None:
        return features
    if len(features.shape) not in {3, 4}:
        raise ValueError('Features dimensions must be 3 or 4 (received dimension %d)' % len(features.shape))
    original_features_shape = features.shape
    if len(features.shape) == 4:
        features = feature_reshape_down(features=features)
    if not isinstance(scalers, list):
        scalers = [scalers]
    if len(scalers) != features.shape[-1]:
        raise ValueError('Number of scalers (%d) does not equal number of feature channels (%d)' % (
            len(scalers), features.shape[-1]))
    for i, scaler in enumerate(scalers):
        features[:, :, i] = scaler.transform(features[:, :, i])

    if len(original_features_shape) == 4:
        return feature_reshape_up(features, num_time_bands=original_features_shape[1],
                                  num_freq_bands=original_features_shape[2], num_channels=original_features_shape[3],
                                  order='C')
    return features


def get_ngram(data: np.ndarray, lookback: int, padding_value=0) -> np.ndarray:
    padding = np.full((lookback,) + data.shape[1:], fill_value=padding_value)
    data_w_padding = np.append(padding, data, axis=0)
    return np.asarray(list(ngrams(data_w_padding, lookback)))


def get_samples_ngram_with_mask(samples: np.ndarray, lookback: int, squeeze: bool = True, reshape: bool = False,
                                sample_padding_value=0, mask_padding_value=1) -> Tuple[np.ndarray, np.ndarray]:
    if reshape:
        ngram_samples = get_ngram(samples.reshape(-1, 1), lookback, padding_value=sample_padding_value)
    else:
        ngram_samples = get_ngram(samples, lookback)
    mask = np.ones((samples.shape[0], 1), dtype=np.int32)
    ngram_mask = get_ngram(mask, lookback, padding_value=mask_padding_value)

    if squeeze:
        ngram_samples = np.squeeze(ngram_samples)
    ngram_mask = np.squeeze(ngram_mask)

    return ngram_samples, ngram_mask


def normalize_tokenized_arrows(arrow_features, arrow_mask):
    arrow_features_max_len = max([len(feature) for feature in arrow_features])
    arrow_mask_max_len = max([len(mask) for mask in arrow_mask])
    max_len = max(arrow_features_max_len, arrow_mask_max_len)
    for i in range(len(arrow_features)):
        feature_len_diff = max_len - len(arrow_features[i])
        mask_len_diff = max_len - len(arrow_mask[i])
        if feature_len_diff == 0 and mask_len_diff > 0:
            arrow_mask[i] = np.concatenate((arrow_mask[i], np.ones((mask_len_diff,))), axis=0)
        elif feature_len_diff > 0 and mask_len_diff == 0:
            arrow_features[i] = np.concatenate((arrow_features[i], np.full((feature_len_diff,), fill_value=0)))
            arrow_mask[i][len(arrow_mask[i]) - feature_len_diff:] = 0
        elif feature_len_diff > 0 and mask_len_diff > 0:
            arrow_features[i] = np.concatenate((arrow_features[i], np.full((feature_len_diff,), fill_value=0)))
            arrow_mask[i] = np.concatenate((arrow_mask[i], np.zeros((mask_len_diff,))), axis=0)
            if feature_len_diff < mask_len_diff:
                arrow_mask[i][:-(mask_len_diff - feature_len_diff)] = 1
            elif feature_len_diff > mask_len_diff:
                arrow_mask[i][len(arrow_mask[i]) - feature_len_diff:] = 0

    return arrow_features, arrow_mask
