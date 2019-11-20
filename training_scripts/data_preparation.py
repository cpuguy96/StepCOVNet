from sklearn.utils import compute_class_weight

import numpy as np
import joblib


def preprocess(features, labels, extra_features=None, sample_weights=None, scalers=None, stride=1):
    import numpy as np
    features_copy = np.copy(features)
    if len(features.shape) > 2:
        if scalers is not None:
            features_copy[:, :, 0] = scalers[0].transform(np.array(features_copy[:, :, 0]))
            features_copy[:, :, 1] = scalers[1].transform(np.array(features_copy[:, :, 1]))
            features_copy[:, :, 2] = scalers[2].transform(np.array(features_copy[:, :, 2]))
        features_copy = featureReshape(features_copy, True, 7)
    else:
        if scalers is not None:
            features_copy = scalers[0].transform(np.array(features_copy))
        features_copy = featureReshape(features_copy, False, 7)
        features_copy = np.expand_dims(np.squeeze(features_copy), axis=1)

    targets = np.copy(labels.reshape(-1, 1))

    if stride > 1:
        if extra_features is not None:
            features_copy = [features_copy[::stride], extra_features[::stride]]
        else:
            features_copy = features_copy[::stride]
        targets = targets[::stride]
        if sample_weights is not None:
            sample_weights = sample_weights[::stride]
    else:
        if extra_features is not None:
            features_copy = [features_copy, extra_features]

    return features_copy, targets, sample_weights


def featureReshape(feature, multi=False, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen * 2 + 1

    feature_reshaped = np.zeros((n_sample, n_row, n_col), dtype='float16')
    if multi:
        feature_reshaped = np.zeros((n_sample, n_row, n_col, feature.shape[-1]), dtype='float16')
    # print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row, n_col), dtype='float16')
        if multi:
            feature_frame = np.zeros((n_row, n_col, feature.shape[-1]), dtype='float16')
        for jj in range(n_col):
            feature_frame[:, jj] = feature[ii][n_row * jj:n_row * (jj + 1)]
        feature_reshaped[ii, :, :] = feature_frame
    return feature_reshaped


def load_data(filename_features,
              filename_extra_features,
              filename_labels,
              filename_sample_weights,
              filename_scalers):

    # load training and validation data
    with open(filename_features, 'rb') as f:
        features = np.load(f)['features']

    if filename_extra_features is not None:
        with open(filename_extra_features, 'rb') as f:
            extra_features = np.load(f)['extra_features']
    else:
        extra_features = None

    with open(filename_labels, 'rb') as f:
        labels = np.load(f)['labels']

    with open(filename_sample_weights, 'rb') as f:
        sample_weights = np.load(f)['sample_weights']

    scaler = []
    for filename_scaler in filename_scalers:
        with open(filename_scaler, 'rb') as f:
            scaler.append(joblib.load(f))

    class_weights = compute_class_weight('balanced', [0, 1], labels)

    class_weights = {0: class_weights[0], 1: class_weights[1]}

    return features, extra_features, labels, sample_weights, class_weights, scaler
