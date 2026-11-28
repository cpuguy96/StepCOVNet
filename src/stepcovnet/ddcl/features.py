"""Beat-aligned audio windows for DDCL placement (`omalley2025ddcl`).

``resample_beat_audio`` follows ``make_onset_feature_context_range`` in
https://github.com/miguelomalley/DDCL/blob/5b1375c642bb708b3c66baf5d880fbf865b85097/util.py
(32 frames linearly spaced between beat start/end on the 10 ms grid).
The underlying 80-band 3-scale log-mel is DDC PRE (`donahue2017ddc`) from
``stepcovnet.ddc.features``.
"""

from __future__ import annotations

import pathlib

import numpy as np

from stepcovnet.ddc import constants as ddc_constants
from stepcovnet.ddc import features as ddc_features
from stepcovnet.ddcl import constants


def resample_beat_audio(
    spec: np.ndarray,
    start_sec: float,
    end_sec: float,
    *,
    n_frames: int = constants.N_FRAMES_PER_BEAT,
    frame_rate: int = constants.FRAME_RATE,
) -> np.ndarray:
    """Sample ``n_frames`` log-mel frames inside ``[start_sec, end_sec)``.

    DDCL converts seconds to 10 ms indices with ``int(time * 100)`` and
    ``np.linspace(..., endpoint=False)``. Out-of-range frames are filled with
    ``log(1e-16)``.

    Args:
        spec: Z-scored or raw log-mel, shape ``(time, n_mels, n_channels)``.
        start_sec: Beat start in seconds.
        end_sec: Beat end in seconds (exclusive).
        n_frames: Frames per beat (paper: 32).
        frame_rate: Mel hop rate (DDC: 100 Hz).

    Returns:
        Array of shape ``(n_frames, n_mels, n_channels)``.

    Raises:
        ValueError: If ``spec`` is not rank-3 or ``n_frames`` is not positive.
    """
    if spec.ndim != 3:
        raise ValueError(f"spec must have rank 3, got shape {spec.shape}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    n_mel_frames = spec.shape[0]
    start_idx = int(float(start_sec) * frame_rate)
    end_idx = int(float(end_sec) * frame_rate)
    if end_idx <= start_idx:
        end_idx = start_idx + 1
    frame_idxs = np.linspace(
        start_idx,
        end_idx,
        num=n_frames,
        endpoint=False,
    ).astype(int)
    fill = np.full(spec.shape[1:], np.log(ddc_constants.LOG_EPS), dtype=spec.dtype)
    out = np.empty((n_frames,) + spec.shape[1:], dtype=spec.dtype)
    for row, frame_idx in enumerate(frame_idxs):
        if 0 <= frame_idx < n_mel_frames:
            out[row] = spec[frame_idx]
        else:
            out[row] = fill
    return out


def beats_to_audio_tensor(
    spec: np.ndarray,
    beat_times: np.ndarray,
    *,
    n_frames: int = constants.N_FRAMES_PER_BEAT,
) -> np.ndarray:
    """Stack per-beat audio tensors for ``n_beats = len(beat_times) - 1``.

    Args:
        spec: Log-mel ``(time, n_mels, n_channels)``.
        beat_times: Start times for beats ``0..n_beats`` (length ``n_beats+1``).
        n_frames: Frames per beat.

    Returns:
        Float32 array ``(n_beats, n_frames, n_mels, n_channels)``.

    Raises:
        ValueError: If ``beat_times`` has fewer than two entries.
    """
    times = np.asarray(beat_times, dtype=np.float64).reshape(-1)
    if times.size < 2:
        raise ValueError("beat_times must include an exclusive end time")
    n_beats = times.size - 1
    stacked = [
        resample_beat_audio(
            spec, float(times[i]), float(times[i + 1]), n_frames=n_frames
        )
        for i in range(n_beats)
    ]
    return np.stack(stacked, axis=0).astype(np.float32)


def zscore_beats(beat_audio: np.ndarray) -> np.ndarray:
    """Z-score over the beat axis, matching DDCL ``models.py`` onset generator.

    Args:
        beat_audio: Shape ``(n_beats, n_frames, n_mels, n_channels)``.

    Returns:
        Normalized copy.

    Raises:
        ValueError: If ``beat_audio`` is not rank-4.
    """
    if beat_audio.ndim != 4:
        raise ValueError(f"beat_audio must have rank 4, got shape {beat_audio.shape}")
    mean = np.mean(beat_audio, axis=0, keepdims=True)
    std = np.std(beat_audio, axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((beat_audio - mean) / std).astype(np.float32)


def causal_windows(
    values: np.ndarray,
    memlen: int,
    *,
    reverse: bool = False,
) -> np.ndarray:
    """Length-``memlen+1`` windows per beat.

    Port of ``windowize`` in DDCL ``util.py``: pad ``memlen`` frames with the
    array minimum (``front_set='min'``). Forward pads the start; reverse pads
    the end.

    Args:
        values: Sequence ``(n_beats, ...)``.
        memlen: Past (or future) beats besides the current one.
        reverse: If True, pad at the end (DDCL ``go_backwards=True``).

    Returns:
        Array ``(n_beats, memlen+1, ...)``.

    Raises:
        ValueError: If ``memlen`` is negative or ``values`` is empty.
    """
    if memlen < 0:
        raise ValueError(f"memlen must be >= 0, got {memlen}")
    if values.shape[0] < 1:
        raise ValueError("values must contain at least one beat")
    pad_shape = (memlen,) + values.shape[1:]
    pad = np.full(pad_shape, np.min(values), dtype=values.dtype)
    n_beats = values.shape[0]
    if reverse:
        padded = np.concatenate([values, pad], axis=0)
    else:
        padded = np.concatenate([pad, values], axis=0)
    windows = np.stack(
        [padded[index : index + memlen + 1] for index in range(n_beats)],
        axis=0,
    )
    return windows


def window_at_beat(
    values: np.ndarray,
    memlen: int,
    beat_idx: int,
    *,
    reverse: bool = False,
    pad_value: float | None = None,
) -> np.ndarray:
    """Return ``causal_windows(...)[beat_idx]`` without allocating all beats.

    Full-split Dataset A cannot keep ``(n_beats, memlen+1, 32, 80, 3)`` for
    every chart in RAM (WSL OOM at ~14 GiB RSS). Training samples one beat
    at a time; eval chunks ``PREDICT_BEAT_BATCH`` beats.

    Args:
        values: Sequence ``(n_beats, ...)``.
        memlen: Past (or future) beats besides the current one.
        beat_idx: Integer beat to window.
        reverse: If True, pad at the end (DDCL ``go_backwards=True``).
        pad_value: Optional constant pad value; defaults to ``np.min(values)``.

    Returns:
        Array ``(memlen+1, ...)``.

    Raises:
        ValueError: If ``memlen`` is negative, ``values`` is empty, or
            ``beat_idx`` is out of range.
    """
    if memlen < 0:
        raise ValueError(f"memlen must be >= 0, got {memlen}")
    n_beats = int(values.shape[0])
    if n_beats < 1:
        raise ValueError("values must contain at least one beat")
    if beat_idx < 0 or beat_idx >= n_beats:
        raise ValueError(f"beat_idx {beat_idx} out of range for {n_beats} beats")
    context = memlen + 1
    fill = np.min(values) if pad_value is None else pad_value
    out = np.empty((context,) + values.shape[1:], dtype=values.dtype)
    if reverse:
        for offset in range(context):
            src = beat_idx + offset
            if src < n_beats:
                out[offset] = values[src]
            else:
                out[offset] = fill
    else:
        for offset in range(context):
            src = beat_idx - memlen + offset
            if src < 0:
                out[offset] = fill
            else:
                out[offset] = values[src]
    return out


def validate_mert_features(
    audio_path: str | pathlib.Path,
    *,
    cache_suffix: str = constants.MERT_CACHE_SUFFIX,
) -> tuple[int, ...]:
    """Validate that a cached MERT ``.npy`` file exists and is 2D/3D via header.

    Uses ``mmap_mode="r"`` so large multi-song splits validate in milliseconds
    without reading feature arrays into RAM.

    Args:
        audio_path: Path to the song audio file; resolves sibling cache file.
        cache_suffix: Sibling filename suffix (default ``.mert.npy``).

    Returns:
        Shape tuple of the cached MERT array.

    Raises:
        FileNotFoundError: If the sibling MERT cache file does not exist.
        ValueError: If the cached array is neither rank-2 nor rank-3.
    """
    mert_path = pathlib.Path(audio_path).with_suffix(cache_suffix)
    if not mert_path.is_file():
        raise FileNotFoundError(f"missing cached MERT features: {mert_path}")
    raw = np.load(mert_path, mmap_mode="r")
    if raw.ndim not in (2, 3):
        raise ValueError(
            f"expected 2D or 3D MERT array in {mert_path}, got {raw.shape}"
        )
    return tuple(int(dim) for dim in raw.shape)


def load_mert_features(
    audio_path: str | pathlib.Path,
    *,
    cache_suffix: str = constants.MERT_CACHE_SUFFIX,
    mmap: bool = False,
) -> np.ndarray:
    """Load precomputed 100 Hz MERT embeddings beside ``audio_path``.

    Supports both 2D ``(time, feature_dim)`` arrays (returned as float32) and
    3D stacked multi-layer ``(time, n_layers, feature_dim)`` arrays (preserving
    float16 when cached in fp16 to bound RAM usage).

    Args:
        audio_path: Path to the song audio file; resolves sibling cache file.
        cache_suffix: Sibling filename suffix (default ``.mert.npy``).
        mmap: If True, open read-only memory-mapped array when dtype already
            matches the target dtype (avoids full-array copy before beat slicing).

    Returns:
        Array of shape ``(time, feature_dim)`` or
        ``(time, n_layers, feature_dim)``.

    Raises:
        FileNotFoundError: If the sibling MERT cache file does not exist.
        ValueError: If the loaded array is neither rank-2 nor rank-3.
    """
    mert_path = pathlib.Path(audio_path).with_suffix(cache_suffix)
    if not mert_path.is_file():
        raise FileNotFoundError(f"missing cached MERT features: {mert_path}")
    raw = np.load(mert_path, mmap_mode="r" if mmap else None)
    if raw.ndim not in (2, 3):
        raise ValueError(
            f"expected 2D or 3D MERT array in {mert_path}, got {raw.shape}"
        )
    target_dtype = (
        np.float16 if (raw.ndim == 3 and raw.dtype == np.float16) else np.float32
    )
    if mmap and raw.dtype == target_dtype:
        return raw
    return np.asarray(raw, dtype=target_dtype)


def resample_beat_mert(
    mert: np.ndarray,
    start_sec: float,
    end_sec: float,
    *,
    n_frames: int = constants.MERT_FRAMES_PER_BEAT,
    frame_rate: int = constants.FRAME_RATE,
) -> np.ndarray:
    """Sample ``n_frames`` MERT embeddings inside ``[start_sec, end_sec)``.

    Args:
        mert: Precomputed 100 Hz MERT embeddings ``(time, feature_dim)`` or
            ``(time, n_layers, feature_dim)``.
        start_sec: Beat start in seconds.
        end_sec: Beat end in seconds (exclusive).
        n_frames: Sub-beat frames to sample (default 16).
        frame_rate: MERT frame rate in Hz (100 Hz).

    Returns:
        Array of shape ``(n_frames, *mert.shape[1:])`` preserving ``mert.dtype``
        with zero padding outside the song bounds.

    Raises:
        ValueError: If ``mert`` is not rank-2/rank-3 or ``n_frames`` is not positive.
    """
    if mert.ndim not in (2, 3):
        raise ValueError(f"mert must have rank 2 or 3, got shape {mert.shape}")
    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    n_mert_frames = mert.shape[0]
    start_idx = int(float(start_sec) * frame_rate)
    end_idx = int(float(end_sec) * frame_rate)
    if end_idx <= start_idx:
        end_idx = start_idx + 1
    frame_idxs = np.linspace(
        start_idx,
        end_idx,
        num=n_frames,
        endpoint=False,
    ).astype(int)
    out_dtype = mert.dtype if mert.dtype in (np.float16, np.float32) else np.float32
    out = np.zeros((n_frames,) + mert.shape[1:], dtype=out_dtype)
    for row, frame_idx in enumerate(frame_idxs):
        if 0 <= frame_idx < n_mert_frames:
            out[row] = mert[frame_idx]
    return out


def beats_to_mert_tensor(
    mert: np.ndarray,
    beat_times: np.ndarray,
    *,
    n_frames: int = constants.MERT_FRAMES_PER_BEAT,
) -> np.ndarray:
    """Stack per-beat MERT tensors for ``n_beats = len(beat_times) - 1``.

    Args:
        mert: Precomputed 100 Hz MERT embeddings ``(time, feature_dim)`` or
            ``(time, n_layers, feature_dim)``.
        beat_times: Start times for beats ``0..n_beats`` (length ``n_beats+1``).
        n_frames: Sub-beat frames per beat.

    Returns:
        Array ``(n_beats, n_frames, *mert.shape[1:])`` preserving fp16/fp32 dtype.

    Raises:
        ValueError: If ``beat_times`` has fewer than two entries.
    """
    times = np.asarray(beat_times, dtype=np.float64).reshape(-1)
    if times.size < 2:
        raise ValueError("beat_times must include an exclusive end time")
    n_beats = times.size - 1
    out_dtype = mert.dtype if mert.dtype in (np.float16, np.float32) else np.float32
    stacked = [
        resample_beat_mert(
            mert, float(times[i]), float(times[i + 1]), n_frames=n_frames
        )
        for i in range(n_beats)
    ]
    return np.stack(stacked, axis=0).astype(out_dtype)


load_or_compute_logmel = ddc_features.load_or_compute_ddc_logmel
