import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(myPath + '/../../'))

from stepcovnet.common.utils import *
from stepcovnet.common.parameters import NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS

TEST_DATA_PATH = os.path.relpath("tests/data/")
TEST_FILE = "scaler.pkl"
TEST_FILE_WITHOUT_EXT = "scaler"
TEST_FILE_WITHOUT_EXT_WITH_RANDOM_CHARS = "/|`:s~;?]@[C(>a!+<#L,)$=e%R^^*"


def test_get_filenames_from_folder():
    file_names = get_filenames_from_folder(TEST_DATA_PATH)
    assert len(file_names) == len(os.listdir(TEST_DATA_PATH))
    assert TEST_FILE in file_names


def test_get_filename_with_ext():
    file_name = get_filename(os.path.join(TEST_DATA_PATH, TEST_FILE), with_ext=True)
    assert file_name == TEST_FILE


def test_get_filename_without_ext():
    file_name = get_filename(os.path.join(TEST_DATA_PATH, TEST_FILE), with_ext=False)
    assert file_name == TEST_FILE_WITHOUT_EXT


def test_standardize_filename():
    standard_file_name = standardize_filename(TEST_FILE_WITHOUT_EXT_WITH_RANDOM_CHARS)
    assert standard_file_name == TEST_FILE_WITHOUT_EXT


def test_get_features_mean_std():
    n_samples = 500
    mean = 0
    std = np.sqrt(1)
    dummy_features = np.random.normal(mean, std, (n_samples, NUM_FREQ_BANDS, NUM_TIME_BANDS))

    mean_std = get_features_mean_std(dummy_features)

    assert len(mean_std[0]) == len(mean_std[1]) == NUM_TIME_BANDS
    assert all(mean - 1e-1 < i < mean + 1e-1 for i in mean_std[0])
    assert all(std - 1e-1 < i < std + 1e-1 for i in mean_std[1])


def test_get_features_mean_std_multi():
    n_samples = 500
    mean = 0
    std = np.sqrt(1)
    dummy_features = np.random.normal(mean, std, (n_samples, NUM_TIME_BANDS, NUM_MULTI_CHANNELS))

    mean_std = get_features_mean_std(dummy_features)

    assert np.array(mean_std).shape == (2, NUM_MULTI_CHANNELS)
    assert all(mean - 1e-1 < i < mean + 1e-1 for i in mean_std[0])
    assert all(std - 1e-1 < i < std + 1e-1 for i in mean_std[1])


def test_get_scalers_not_reshaped():
    n_samples = 10
    mean = 0
    std = np.sqrt(1)
    # dummy_features = np.random.normal(mean, std, (n_samples, NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS))

    flat_features = np.arange(NUM_FREQ_BANDS * NUM_TIME_BANDS)
    flat_features = np.concatenate([[flat_features]] * n_samples, axis=0)

    try:
        get_scalers(flat_features, multi=False)
    except Exception as ex:
        assert isinstance(ex, ValueError)
        return
    assert 0  # failed to check if reshaped


def test_get_scalers_not_reshaped_multi():
    n_samples = 10
    flat_features = np.arange(NUM_FREQ_BANDS * NUM_TIME_BANDS * NUM_MULTI_CHANNELS)
    flat_features = np.concatenate([[flat_features]] * n_samples, axis=0)

    try:
        get_scalers(flat_features, multi=False)
    except Exception as ex:
        assert isinstance(ex, ValueError)
        return
    assert 0  # failed to check if reshaped


def test_get_scalers_reshaped():
    n_samples = 10
    mean = 0
    std = np.sqrt(1)
    dummy_features = np.random.normal(mean, std, (n_samples, NUM_FREQ_BANDS, NUM_TIME_BANDS))

    scalers = get_scalers(dummy_features, multi=False)

    assert scalers.shape == (2, NUM_FREQ_BANDS)
    assert all(mean - 5e-1 < i < mean + 5e-1 for i in scalers[0])
    assert all(std - 5e-1 < i < std + 5e-1 for i in scalers[1])


def test_get_scalers_reshaped_multi():
    n_samples = 10
    mean = 0
    std = np.sqrt(1)
    dummy_features = np.random.normal(mean, std, (n_samples, NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS))

    scalers = get_scalers(dummy_features, multi=True)

    assert scalers.shape == (NUM_MULTI_CHANNELS, 2, NUM_FREQ_BANDS)
    for scaler in scalers:
        assert all(mean - 5e-1 < i < mean + 5e-1 for i in scaler[0])
        assert all(std - 5e-1 < i < std + 5e-1 for i in scaler[1])


def test_feature_reshape():
    n_samples = 10
    n_rows = NUM_FREQ_BANDS
    n_cols = NUM_TIME_BANDS

    dummy_features = np.arange(n_rows * n_cols).reshape((n_rows, n_cols), order='F')
    dummy_features = np.concatenate([[dummy_features]] * n_samples, axis=0)

    flat_features = np.arange(n_rows * n_cols)
    flat_features = np.concatenate([[flat_features]] * n_samples, axis=0)

    reshaped_features = feature_reshape(flat_features, multi=False)

    assert reshaped_features.shape == dummy_features.shape
    assert np.array_equal(reshaped_features, dummy_features)


def test_feature_reshape_multi():
    n_samples = 10
    n_rows = NUM_FREQ_BANDS
    n_cols = NUM_TIME_BANDS

    nd_a_array = np.arange(n_rows * n_cols * NUM_MULTI_CHANNELS).reshape((n_rows, n_cols, NUM_MULTI_CHANNELS),
                                                                         order='F')
    dummy_features = np.concatenate([[nd_a_array]] * n_samples, axis=0)

    flat_dummy = np.arange(n_rows * n_cols * NUM_MULTI_CHANNELS).reshape((n_rows * n_cols, NUM_MULTI_CHANNELS),
                                                                         order='F')
    flat_dummy = np.concatenate([[flat_dummy]] * n_samples, axis=0)

    reshaped_features = feature_reshape(flat_dummy, multi=True)

    assert reshaped_features.shape == dummy_features.shape
    assert np.array_equal(reshaped_features, dummy_features)


def test_pre_process():
    n_samples = 10
    mean = 0
    std = np.sqrt(1)
    dummy_features = np.random.normal(mean, std, (n_samples, NUM_FREQ_BANDS, NUM_TIME_BANDS))

    processed = pre_process(dummy_features, multi=False)

    assert processed.shape == np.expand_dims(dummy_features, axis=1).shape


def test_pre_process_multi():
    n_samples = 10
    mean = 0
    std = np.sqrt(1)
    dummy_features = np.random.normal(mean, std, (n_samples, NUM_FREQ_BANDS, NUM_TIME_BANDS, NUM_MULTI_CHANNELS))

    processed = pre_process(dummy_features, multi=True)

    assert processed.shape == dummy_features.shape
