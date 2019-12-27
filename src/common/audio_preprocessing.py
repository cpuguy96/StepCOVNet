import librosa
import numpy as np
from madmom.audio.filters import MelFilterbank
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
from madmom.processors import SequentialProcessor, ParallelProcessor

from common.Fprev_sub import Fprev_sub


def get_feature_processors(sample_rate, hopsize_t, frame_size):
    frames = FramedSignalProcessor(frame_size=frame_size, hopsize=int(sample_rate * hopsize_t))
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(
        filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
        norm_filters=True, unique_filters=False)
    spec = LogarithmicSpectrogramProcessor(log=np.log, add=np.spacing(1))
    return SequentialProcessor([frames, stft, filt, spec])


def get_madmom_melbank(file_name, sample_rate, hopsize_t, multi):
    sig = SignalProcessor(num_channels=1, sample_rate=sample_rate)
    if multi:
        multi_proc = ParallelProcessor([get_feature_processors(sample_rate, hopsize_t, frame_size)
                                        for frame_size in [1024, 2048, 4096]])
        return SequentialProcessor([sig, multi_proc, np.dstack])(file_name)
    else:
        frame_size = 2048
        single_proc = get_feature_processors(sample_rate, hopsize_t, frame_size)
        return SequentialProcessor([sig, single_proc])(file_name)


def nbf_2D(mfcc, nlen):
    mfcc = np.array(mfcc).transpose()
    mfcc_out = np.array(mfcc, copy=True)
    for ii in range(1, nlen + 1):
        mfcc_right_shift = Fprev_sub(mfcc, w=ii)
        mfcc_left_shift = Fprev_sub(mfcc, w=-ii)
        mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
    feature = mfcc_out.transpose()
    return feature


def get_madmom_log_mels(file_name, sample_rate, hopsize_t, multi):
    mfcc = get_madmom_melbank(file_name, sample_rate, hopsize_t, multi)
    if multi:
        mfcc_conc = [nbf_2D(mfcc[:, :, i], 7) for i in range(3)]
        return np.stack(mfcc_conc, axis=2)
    else:
        return nbf_2D(mfcc, 7)


def get_librosa_features(file_name, sample_rate, hopsize_t):
    samples, _ = librosa.load(file_name, sr=sample_rate)
    onset_times = librosa.onset.onset_detect(y=samples,
                                             sr=sample_rate,
                                             units="time",
                                             hop_length=int(sample_rate * hopsize_t))
    return np.array(np.around(np.array(onset_times) / hopsize_t), dtype=int)


def get_madmom_features(file_name, hopsize_t):
    proc = DBNBeatTrackingProcessor(max_bpm=300,
                                    fps=int(1 / hopsize_t))
    act = RNNBeatProcessor()
    pre_processor = SequentialProcessor([act, proc])
    beat_times = pre_processor(file_name)
    return np.array(np.around(np.array(beat_times) / hopsize_t), dtype=int)


def get_madmom_librosa_features(file_name, sample_rate, hopsize_t, num_frames, frame_start=0):
    # TODO: add ability to choose which features to add
    # librosa features
    librosa_frames = get_librosa_features(file_name, sample_rate, hopsize_t)

    # madmom features
    madmom_frames = get_madmom_features(file_name, hopsize_t)

    # fill in blanks and return
    librosa_features = np.zeros((num_frames,))
    madmom_features = np.zeros((num_frames,))

    librosa_features[librosa_frames - frame_start] = 1
    madmom_features[madmom_frames - frame_start] = 1

    return np.hstack((librosa_features.reshape(-1, 1), madmom_features.reshape(-1, 1)))
