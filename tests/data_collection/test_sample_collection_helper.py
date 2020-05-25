from stepcovnet.data_collection.sample_collection_helper import *

import os

TEST_DATA_PATH = os.path.relpath("tst/data/")
TEST_TXT_FILE = "zombie_maker.txt"
TEST_FILE_NAME = "zombie_maker"

import pytest
@pytest.fixture()
def mock_get_madmom_log_mels(monkeypatch):
    monkeypatch.setattr(data_collection.sample_collection_helper, "get_madmom_log_mels", )


def test_remove_out_of_range():
    frames = np.array([-1e8, -1, 0, 1, 50, 100, 1000, 1e8])
    frame_start = 0
    frames_end = 500
    removed_frames = remove_out_of_range(frames, frame_start, frames_end)
    assert set(frames).intersection(removed_frames) == {0, 1, 50, 100}


def test_timings_parser():
    timings = timings_parser(os.path.join(TEST_DATA_PATH, TEST_TXT_FILE))
    assert [float(timing) for timing in timings]


def test_dump_feature_onset_helper(monkeypatch):
    num_frames = 500
    fake_log_mels = np.ones((num_frames, 80 * 15))
    fake_timings = [0.1, 0.2, 5, 5.1, 10, 11.111, 44]

    def mock_log_mels():
        return fake_log_mels

    def mock_timings():
        return fake_timings



    monkeypatch.setattr(data_collection.sample_collection_helper, "timings_parser", mock_timings)

    mfcc, frames_onset, frame_start, frame_end = dump_feature_onset_helper(wav_path=TEST_DATA_PATH,
                                                                           timing_path=TEST_DATA_PATH,
                                                                           file_name=TEST_FILE_NAME,
                                                                           multi=False)

    assert frame_end == num_frames - 1
    assert len(frames_onset) == fake_timings
    assert frames_onset == frames_onset.astype("int8")
