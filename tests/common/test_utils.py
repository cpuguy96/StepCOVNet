from stepcovnet.common.utils import *

from stepcovnet.configuration.parameters import NUM_FREQ_BANDS, NUM_TIME_BANDS

DATA_RELATIVE_PATH = "tests/data/"
TEST_FILE = "scaler.pkl"
TEST_FILE_WITHOUT_EXT = "scaler"
TEST_FILE_WITHOUT_EXT_WITH_RANDOM_CHARS = "/|`:s~;?]@[C(>a!+<#L,)$=e%R^^*"
TEST_DATA_PATH = os.path.relpath(os.path.join(DATA_RELATIVE_PATH))


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


def test_feature_reshape():
    n_samples = 500
    n_rows = NUM_FREQ_BANDS
    n_cols = NUM_TIME_BANDS
    dummy_features = np.empty((n_samples, n_rows * n_cols))
    reshaped_features = feature_reshape(dummy_features, multi=False)
    assert reshaped_features.shape == (n_samples, n_rows, n_cols)


def test_feature_reshape_multi():
    n_samples = 500
    n_rows = NUM_FREQ_BANDS
    n_cols = NUM_TIME_BANDS
    dummy_features = np.empty((n_samples, n_rows * n_cols, 3))
    reshaped_features = feature_reshape(dummy_features, multi=True)
    assert reshaped_features.shape == (n_samples, n_rows, n_cols, 3)
