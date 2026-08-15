"""Multi-scale 80-band log-mel PRE for DDC placement (`donahue2017ddc`)."""

from __future__ import annotations

import pathlib

import librosa
import numpy as np
from numpy.lib import stride_tricks

from stepcovnet.ddc import constants


def difficulty_one_hot(label: str) -> np.ndarray:
    """Return a length-5 one-hot vector for a standard DDR difficulty.

    Args:
        label: Lowercase difficulty name (``beginner`` … ``challenge``).

    Returns:
        Float32 vector of shape ``(N_DIFFICULTIES,)``.

    Raises:
        ValueError: If ``label`` is not a standard DDC difficulty.
    """
    key = str(label).strip().lower()
    try:
        index = constants.DIFFICULTY_LABELS.index(key)
    except ValueError as exc:
        raise ValueError(
            f"unsupported DDC difficulty {label!r}; "
            f"expected one of {constants.DIFFICULTY_LABELS}"
        ) from exc
    one_hot = np.zeros((constants.N_DIFFICULTIES,), dtype=np.float32)
    one_hot[index] = 1.0
    return one_hot


def difficulty_index(label: str) -> int:
    """Return the integer id for a standard DDR difficulty.

    Args:
        label: Lowercase difficulty name.

    Returns:
        Index into ``DIFFICULTY_LABELS``.

    Raises:
        ValueError: If ``label`` is not a standard DDC difficulty.
    """
    key = str(label).strip().lower()
    try:
        return constants.DIFFICULTY_LABELS.index(key)
    except ValueError as exc:
        raise ValueError(
            f"unsupported DDC difficulty {label!r}; "
            f"expected one of {constants.DIFFICULTY_LABELS}"
        ) from exc


def load_mono_audio(audio_path: str | pathlib.Path) -> np.ndarray:
    """Load mono audio at 44.1 kHz.

    Args:
        audio_path: Path to an audio file.

    Returns:
        Float32 waveform with shape ``(num_samples,)``.
    """
    y, _sr = librosa.load(str(audio_path), sr=constants.SAMPLE_RATE, mono=True)
    return np.asarray(y, dtype=np.float32)


def audio_to_ddc_logmel(audio: np.ndarray) -> np.ndarray:
    """Compute 80-band log-mel at 23/46/93 ms windows, 10 ms hop.

    Uses Librosa STFT sizes 1024/2048/4096 at 44.1 kHz. DDC's original extractors
    used Essentia; this is the cited PRE recipe (`schluter2014onset`,
    `hamel2012multiscale`, `donahue2017ddc`), not a bit-identical port.

    Args:
        audio: Mono waveform at ``SAMPLE_RATE``.

    Returns:
        Array of shape ``(time, 80, 3)`` (log magnitude, not yet z-scored).
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    channels = []
    n_frames = None
    for n_fft in constants.FFT_SIZES:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=constants.SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=constants.HOP_LENGTH,
            win_length=n_fft,
            n_mels=constants.N_MELS,
            center=True,
            power=2.0,
        )
        log_mel = np.log(mel + constants.LOG_EPS)
        if n_frames is None:
            n_frames = log_mel.shape[1]
        else:
            log_mel = log_mel[:, :n_frames]
        channels.append(log_mel.T.astype(np.float32))
    stacked = np.stack(channels, axis=-1)
    return stacked.astype(np.float32)


def zscore_bands(spec: np.ndarray) -> np.ndarray:
    """Standardize each frequency band and FFT scale across time.

    Args:
        spec: Log-mel array of shape ``(time, n_mels, n_channels)``.

    Returns:
        Z-scored array of the same shape and dtype float32.
    """
    mean = np.mean(spec, axis=0, keepdims=True)
    std = np.std(spec, axis=0, keepdims=True)
    return ((spec - mean) / (std + 1e-6)).astype(np.float32)


def context_windows(
    spec: np.ndarray,
    *,
    radius: int = constants.CONTEXT_RADIUS,
) -> np.ndarray:
    """Stack ±``radius`` frames around each time step.

    Args:
        spec: Array of shape ``(time, n_mels, n_channels)``.
        radius: Past/future frames to include (DDC uses 7 → 15-frame windows).

    Returns:
        Array of shape ``(time, 2*radius+1, n_mels, n_channels)``.

    Raises:
        ValueError: If ``radius`` is negative or ``spec`` is not rank 3.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if spec.ndim != 3:
        raise ValueError(f"spec must have rank 3, got shape {spec.shape}")
    window = radius * 2 + 1
    padded = np.pad(spec, ((radius, radius), (0, 0), (0, 0)), mode="constant")
    view = stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.moveaxis(view, -1, 1).astype(np.float32, copy=False)


def context_windows_span(
    spec: np.ndarray,
    start: int,
    length: int,
    *,
    radius: int = constants.CONTEXT_RADIUS,
) -> np.ndarray:
    """Return ±``radius`` context windows for frames ``[start, start+length)``.

    Args:
        spec: Array of shape ``(time, n_mels, n_channels)``.
        start: Inclusive start frame.
        length: Number of frames to cover.
        radius: Past/future frames to include (DDC uses 7).

    Returns:
        Array of shape ``(length, 2*radius+1, n_mels, n_channels)``.

    Raises:
        ValueError: If ``start``/``length`` are out of range or ``radius`` is
            negative.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")
    if spec.ndim != 3:
        raise ValueError(f"spec must have rank 3, got shape {spec.shape}")
    n_frames = int(spec.shape[0])
    if start < 0 or length < 1 or start + length > n_frames:
        raise ValueError(f"span [{start}, {start + length}) is outside [0, {n_frames})")
    window = radius * 2 + 1
    padded = np.pad(spec, ((radius, radius), (0, 0), (0, 0)), mode="constant")
    region = padded[start : start + length + window - 1]
    view = stride_tricks.sliding_window_view(region, window, axis=0)
    return np.moveaxis(view, -1, 1).astype(np.float32, copy=False)


def feature_cache_path(audio_path: str | pathlib.Path) -> pathlib.Path:
    """Return the on-disk cache path for a DDC log-mel array.

    Args:
        audio_path: Path to the source audio file.

    Returns:
        Sibling ``*.ddc_mel.npy`` path.
    """
    path = pathlib.Path(audio_path)
    return path.with_name(path.stem + constants.FEATURE_CACHE_SUFFIX)


def load_or_compute_ddc_logmel(
    audio_path: str | pathlib.Path,
    *,
    cache: bool = True,
) -> np.ndarray:
    """Load cached DDC log-mel features or compute and optionally save them.

    Args:
        audio_path: Path to the source audio file.
        cache: When True, read/write ``feature_cache_path``.

    Returns:
        Z-scored log-mel array of shape ``(time, 80, 3)``.
    """
    cache_path = feature_cache_path(audio_path)
    if cache and cache_path.is_file():
        loaded = np.load(cache_path)
        return np.asarray(loaded, dtype=np.float32)
    spec = zscore_bands(audio_to_ddc_logmel(load_mono_audio(audio_path)))
    if cache:
        np.save(cache_path, spec)
    return spec


def times_to_frame_target(
    times_sec: np.ndarray,
    n_frames: int,
    *,
    hop_sec: float = constants.HOP_SEC,
) -> np.ndarray:
    """Rasterize onset times onto the 10 ms DDC grid.

    Args:
        times_sec: Onset times in seconds.
        n_frames: Number of frames in the spectrogram.
        hop_sec: Frame hop in seconds.

    Returns:
        Float32 vector of shape ``(n_frames,)`` with 1.0 at onset frames.
    """
    target = np.zeros((n_frames,), dtype=np.float32)
    if n_frames <= 0:
        return target
    for time_sec in np.asarray(times_sec, dtype=np.float64).reshape(-1):
        frame_idx = int(round(float(time_sec) / hop_sec))
        if 0 <= frame_idx < n_frames:
            target[frame_idx] = 1.0
    return target
