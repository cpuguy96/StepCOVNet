"""Beat-aligned audio windows for DDCL placement (`omalley2025ddcl`).

``resample_beat_audio`` follows ``make_onset_feature_context_range`` in
https://github.com/miguelomalley/DDCL/blob/5b1375c642bb708b3c66baf5d880fbf865b85097/util.py
(32 frames linearly spaced between beat start/end on the 10 ms grid).
The underlying 80-band 3-scale log-mel is DDC PRE (`donahue2017ddc`) from
``stepcovnet.ddc.features``.
"""

from __future__ import annotations

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
    pad_value = np.min(values)
    out = np.empty((context,) + values.shape[1:], dtype=values.dtype)
    if reverse:
        for offset in range(context):
            src = beat_idx + offset
            if src < n_beats:
                out[offset] = values[src]
            else:
                out[offset] = pad_value
    else:
        for offset in range(context):
            src = beat_idx - memlen + offset
            if src < 0:
                out[offset] = pad_value
            else:
                out[offset] = values[src]
    return out


load_or_compute_logmel = ddc_features.load_or_compute_ddc_logmel
