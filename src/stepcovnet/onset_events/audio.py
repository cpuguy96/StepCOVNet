"""Waveform loading, truncation, and padding for event-based onset detection."""

import librosa
import numpy as np

from stepcovnet import constants

DEFAULT_MAX_AUDIO_SECONDS = 300


def max_samples_for_cap(
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
    sample_rate: int = constants.TARGET_SR,
) -> int:
    """Return the sample count for an audio duration cap.

    Args:
        max_audio_seconds: Maximum audio duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        ``int(max_audio_seconds * sample_rate)``.
    """
    return int(max_audio_seconds * sample_rate)


DEFAULT_MAX_SAMPLES = max_samples_for_cap()


def load_waveform(
    audio_path: str,
    target_sample_rate: int = constants.TARGET_SR,
) -> np.ndarray:
    """Load a mono waveform from an audio file.

    Uses librosa at ``target_sample_rate`` (default ``constants.TARGET_SR``,
    44100 Hz) and peak-normalizes like the dense onset audio path.

    Args:
        audio_path: Path to an audio file readable by librosa.
        target_sample_rate: Target sample rate in Hz.

    Returns:
        One-dimensional float32 waveform.
    """
    y, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
    if sr != target_sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return np.asarray(y, dtype=np.float32)


def truncate_waveform(waveform: np.ndarray, max_samples: int) -> np.ndarray:
    """Keep at most the first ``max_samples`` of a waveform.

    Args:
        waveform: Input waveform.
        max_samples: Maximum number of samples to retain.

    Returns:
        Float32 array with length ``min(len(waveform), max_samples)``.
    """
    n = min(int(waveform.size), max_samples)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(waveform[:n], dtype=np.float32)


def pad_waveform(waveform: np.ndarray, max_samples: int) -> np.ndarray:
    """Zero-pad a waveform on the right to ``max_samples`` length.

    Args:
        waveform: Input waveform with length at most ``max_samples``.
        max_samples: Target length in samples.

    Returns:
        Float32 array of shape ``(max_samples,)``.

    Raises:
        ValueError: If ``len(waveform)`` exceeds ``max_samples``.
    """
    n = int(waveform.size)
    if n > max_samples:
        raise ValueError(
            f"waveform length {n} exceeds max_samples {max_samples}; truncate first"
        )
    out = np.zeros(max_samples, dtype=np.float32)
    if n > 0:
        out[:n] = np.asarray(waveform, dtype=np.float32)
    return out
