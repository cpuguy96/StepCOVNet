"""Mel STFT helpers for onset models (shared by datasets and ssl_features)."""

from __future__ import annotations

import librosa
import numpy as np

from stepcovnet import constants

_TARGET_SR = constants.TARGET_SR
_HOP_COEFF = constants.HOP_COEFF
_N_MELS = constants.N_MELS
_F_MIN = 27.5
_F_MAX = 12000
_WIN_COEFF = 0.025


def audio_to_spectrogram(audio_path: str) -> np.ndarray:
    """Convert an audio file to a mel-spectrogram using LibROSA.

    Args:
        audio_path: Path to the audio file (mp3, ogg, or wav).

    Returns:
        A 2D numpy array representing the mel-spectrogram in decibels,
        with shape (n_mels, time_steps).
    """
    y, sr = librosa.load(audio_path, sr=_TARGET_SR)

    if sr != _TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_TARGET_SR)

    y = y / np.max(np.abs(y))

    hop_length = int(round(_TARGET_SR * _HOP_COEFF))
    win_length = int(round(_TARGET_SR * _WIN_COEFF))
    n_fft = 2 ** int(np.ceil(np.log(win_length) / np.log(2.0)))

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=_TARGET_SR,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        fmin=_F_MIN,
        fmax=_F_MAX,
        n_mels=_N_MELS,
    )

    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def onset_frame_count(audio_path: str) -> int:
    """Return the onset model time-step count for an audio file (mel STFT grid).

    Args:
        audio_path: Path to the audio file.

    Returns:
        Number of time steps along the onset detection grid for this file.
    """
    return int(audio_to_spectrogram(audio_path).shape[1])
