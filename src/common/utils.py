import re
from os import listdir
from os.path import join, isfile, splitext, basename

import numpy as np


def get_filenames_from_folder(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def get_filename(file_path, with_ext=True):
    if with_ext:
        return basename(file_path)
    else:
        return splitext(basename(file_path))[0]


def standardize_filename(filename):
    return re.sub("[^a-z0-9-_]", "", filename.lower())


def feature_reshape(feature, multi=False, nlen=10):
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
