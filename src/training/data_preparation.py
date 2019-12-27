import pickle

import joblib
import numpy as np
from sklearn.utils import compute_class_weight

from common.utils import feature_reshape


def preprocess(features, labels, extra_features=None, sample_weights=None, scalers=None):
    features_copy = np.copy(features)
    if len(features.shape) > 2:
        if scalers is not None:
            features_copy[:, :, 0] = scalers[0].transform(np.array(features_copy[:, :, 0]))
            features_copy[:, :, 1] = scalers[1].transform(np.array(features_copy[:, :, 1]))
            features_copy[:, :, 2] = scalers[2].transform(np.array(features_copy[:, :, 2]))
        features_copy = feature_reshape(features_copy, True, 7)
    else:
        if scalers is not None:
            features_copy = scalers[0].transform(np.array(features_copy))
        features_copy = feature_reshape(features_copy, False, 7)
        features_copy = np.expand_dims(np.squeeze(features_copy), axis=1)

    targets = np.copy(labels.reshape(-1, 1))

    if extra_features is not None:
        features_copy = [features_copy, extra_features]

    return features_copy, targets, sample_weights


def load_data(filename_features,
              filename_extra_features,
              filename_labels,
              filename_sample_weights,
              filename_scalers,
              filename_pretrained_model):
    # load training and validation data
    with open(filename_features, 'rb') as f:
        features = joblib.load(f)

    if filename_extra_features is not None:
        with open(filename_extra_features, 'rb') as f:
            extra_features = joblib.load(f)
    else:
        extra_features = None

    with open(filename_labels, 'rb') as f:
        labels = joblib.load(f)

    with open(filename_sample_weights, 'rb') as f:
        sample_weights = joblib.load(f)

    scaler = []
    for filename_scaler in filename_scalers:
        with open(filename_scaler, 'rb') as f:
            scaler.append(pickle.load(f))

    if filename_pretrained_model is not None:
        from tensorflow.keras.models import load_model
        pretrained_model = load_model(filename_pretrained_model, compile=False)
    else:
        pretrained_model = None

    class_weights = compute_class_weight('balanced', [0, 1], labels)

    class_weights = {0: class_weights[0], 1: class_weights[1]}

    return features, extra_features, labels, sample_weights, class_weights, scaler, pretrained_model
