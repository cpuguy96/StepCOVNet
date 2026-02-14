import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(myPath + "/../../"))

from legecy_v1.stepcovnet import utils
from legecy_v1.stepcovnet.constants import NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS

TEST_DATA_PATH = os.path.relpath("tests/data/")
TEST_FILE = "scaler.pkl"
TEST_FILE_WITHOUT_EXT = "scaler"
TEST_FILE_WITHOUT_EXT_WITH_RANDOM_CHARS = "/|`:s~;?]@[C(>a!+<#L,)$=e%R^^*"


def test_get_filenames_from_folder():
    file_names = utils.get_filenames_from_folder(TEST_DATA_PATH)
    assert len(file_names) == len(os.listdir(TEST_DATA_PATH))
    assert TEST_FILE in file_names


def test_get_filename_with_ext():
    file_name = utils.get_filename(os.path.join(TEST_DATA_PATH, TEST_FILE), with_ext=True)
    assert file_name == TEST_FILE


def test_get_filename_without_ext():
    file_name = utils.get_filename(os.path.join(TEST_DATA_PATH, TEST_FILE), with_ext=False)
    assert file_name == TEST_FILE_WITHOUT_EXT


def test_standardize_filename():
    standard_file_name = utils.standardize_filename(TEST_FILE_WITHOUT_EXT_WITH_RANDOM_CHARS)
    assert standard_file_name == TEST_FILE_WITHOUT_EXT


def test_feature_reshape():
    n_samples = 10
    n_rows = NUM_TIME_BANDS
    n_cols = NUM_FREQ_BANDS

    dummy_features = np.arange(n_rows * n_cols).reshape((n_rows, n_cols, 1), order="F")
    dummy_features = np.concatenate([[dummy_features]] * n_samples, axis=0)

    flat_features = np.arange(n_rows * n_cols)
    flat_features = np.concatenate([[flat_features]] * n_samples, axis=0)

    reshaped_features = utils.feature_reshape_up(
        flat_features,
        num_freq_bands=NUM_FREQ_BANDS,
        num_time_bands=NUM_TIME_BANDS,
        num_channels=1,
    )

    assert reshaped_features.shape == dummy_features.shape
    assert np.array_equal(reshaped_features, dummy_features)


def test_feature_reshape_multi():
    n_samples = 10
    n_rows = NUM_TIME_BANDS
    n_cols = NUM_FREQ_BANDS

    nd_a_array = np.arange(n_rows * n_cols * NUM_MULTI_CHANNELS).reshape(
        (n_rows, n_cols, NUM_MULTI_CHANNELS), order="F"
    )
    dummy_features = np.concatenate([[nd_a_array]] * n_samples, axis=0)

    flat_dummy = np.arange(n_rows * n_cols * NUM_MULTI_CHANNELS).reshape(
        (n_rows * n_cols, NUM_MULTI_CHANNELS), order="F"
    )
    flat_dummy = np.concatenate([[flat_dummy]] * n_samples, axis=0)

    reshaped_features = utils.feature_reshape_up(
        flat_dummy,
        num_freq_bands=NUM_FREQ_BANDS,
        num_time_bands=NUM_TIME_BANDS,
        num_channels=NUM_MULTI_CHANNELS,
    )

    assert reshaped_features.shape == dummy_features.shape
    assert np.array_equal(reshaped_features, dummy_features)
