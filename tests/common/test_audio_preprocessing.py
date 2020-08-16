import os

from stepcovnet.common.audio_preprocessing import *
from stepcovnet.common.constants import HOPSIZE_T
from stepcovnet.common.constants import NUM_FREQ_BANDS
from stepcovnet.common.constants import NUM_MULTI_CHANNELS
from stepcovnet.common.constants import NUM_TIME_BANDS
from stepcovnet.common.constants import SAMPLE_RATE
from stepcovnet.common.parameters import CONFIG

TEST_DATA_PATH = os.path.relpath("tests/data/")
TEST_FILE = "tide.wav"


def test_get_feature_processors():
    feature_processors = get_feature_processors(SAMPLE_RATE, HOPSIZE_T, SINGLE_CHANNEL_FRAME_SIZE, config=CONFIG)
    assert isinstance(feature_processors, SequentialProcessor)
    assert len(feature_processors) == 4
    for processor in feature_processors:
        assert isinstance(processor, (FramedSignalProcessor, ShortTimeFourierTransformProcessor,
                                      FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor))


def test_get_madmom_log_mels():
    log_mels = get_madmom_log_mels(os.path.join(TEST_DATA_PATH, TEST_FILE),
                                   config=CONFIG,
                                   multi=False)
    assert len(log_mels.shape) == 2
    assert log_mels.shape[1] == NUM_FREQ_BANDS * NUM_TIME_BANDS


def test_get_madmom_log_mels_multi():
    multi_log_mels = get_madmom_log_mels(os.path.join(TEST_DATA_PATH, TEST_FILE),
                                         config=CONFIG,
                                         multi=True)
    assert len(multi_log_mels.shape) == 3
    assert multi_log_mels.shape[1] == NUM_FREQ_BANDS * NUM_TIME_BANDS
    assert multi_log_mels.shape[2] == NUM_MULTI_CHANNELS


def test_get_librosa_frames():
    librosa_frames = get_librosa_frames(os.path.join(TEST_DATA_PATH, TEST_FILE),
                                        SAMPLE_RATE, HOPSIZE_T)
    assert [int(frame) for frame in librosa_frames] == list(librosa_frames)
    assert sum(frame < 0 for frame in librosa_frames) == 0


def test_get_madmom_frames():
    madmom_frames = get_librosa_frames(os.path.join(TEST_DATA_PATH, TEST_FILE),
                                       SAMPLE_RATE, HOPSIZE_T)
    assert [int(frame) for frame in madmom_frames] == list(madmom_frames)
    assert sum(frame < 0 for frame in madmom_frames) == 0


def get_madmom_librosa_features(mocker):
    num_frames = 500
    librosa_frame_timings = np.random.randint(num_frames, size=num_frames)
    madmom_frame_timings = np.random.randint(num_frames, size=num_frames)

    mocker.patch("stepmania.common.audio_preprocessing.get_librosa_frames", return_value=librosa_frame_timings)
    mocker.patch("stepmania.common.audio_preprocessing.get_madmom_frames", return_value=madmom_frame_timings)

    extra_features = get_madmom_librosa_features(os.path.join(TEST_DATA_PATH, TEST_FILE),
                                                 SAMPLE_RATE, HOPSIZE_T, num_frames)

    assert extra_features.shape == (num_frames, 2)
    assert all(i == 1 for i in extra_features[librosa_frame_timings, 0])
    assert all(i == 1 for i in extra_features[madmom_frame_timings, 1])
