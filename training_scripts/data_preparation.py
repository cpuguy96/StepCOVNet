import pickle
import numpy as np
from sklearn.utils import compute_class_weight


def featureReshape(feature, multi=False, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen * 2 + 1

    feature_reshaped = np.zeros((n_sample, n_row, n_col), dtype='float32')
    if multi:
        feature_reshaped = np.zeros((n_sample, n_row, n_col, feature.shape[-1]), dtype='float32')
    # print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row, n_col), dtype='float32')
        if multi:
            feature_frame = np.zeros((n_row, n_col, feature.shape[-1]), dtype='float32')
        for jj in range(n_col):
            feature_frame[:, jj] = feature[ii][n_row * jj:n_row * (jj + 1)]
        feature_reshaped[ii, :, :] = feature_frame
    return feature_reshaped


def load_data(filename_labels_train_validation_set,
              filename_sample_weights,
              filename_scalers):

    # load training and validation data

    with open(filename_labels_train_validation_set, 'rb') as f:
        Y_train_validation = pickle.load(f)

    with open(filename_sample_weights, 'rb') as f:
        sample_weights = pickle.load(f)

    scaler = []
    for filename_scaler in filename_scalers:
        with open(filename_scaler, 'rb') as f:
            scaler.append(pickle.load(f))

    # this is the filename indices
    indices_train_validation = range(len(Y_train_validation))

    class_weights = compute_class_weight('balanced', [0, 1], Y_train_validation)

    class_weights = {0: class_weights[0], 1: class_weights[1]}

    return indices_train_validation, Y_train_validation, sample_weights, class_weights, scaler
