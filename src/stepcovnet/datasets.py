"""Data collection and preprocessing for StepCOVNet.

This module provides functionality to load audio and StepMania chart files,
process them into spectrograms and target vectors, and create PyTorch
datasets and data loaders for training.
"""

from __future__ import annotations

import pathlib
from typing import Any, cast

import librosa
import numpy as np
import torch
from scipy import interpolate
from torch.utils.data import DataLoader, Dataset

from stepcovnet import config, constants, mel_onset, pairing, ssl_features
from stepcovnet.dataset_prep import training_index, training_loader

HOP_COEFF = constants.HOP_COEFF

_DIFFICULTY_MAP = {"beginner": 0, "easy": 1, "medium": 2, "hard": 3, "challenge": 4}
_N_MELS = constants.N_MELS
_N_TARGET = 1
_F_MIN = 27.5
_F_MAX = 12000
_WIN_COEFF = 0.025
_TARGET_SR = constants.TARGET_SR


def normalize_onset_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Normalize a mel spectrogram for onset model input.

    Uses per-mel-bin normalization (mean and std across time). Training and
    inference always use this normalization; it is critical for consistent
    results.

    Args:
        spec: Spectrogram with shape (time_steps, n_mels), e.g. from
            audio_to_spectrogram(audio_path).T.

    Returns:
        Normalized spectrogram with same shape, dtype float32.
    """
    mean = np.mean(spec, axis=0, keepdims=True)
    std = np.std(spec, axis=0, keepdims=True)
    return ((spec - mean) / (std + 1e-6)).astype(np.float32)


def _base4_to_int(base4_string: str) -> int:
    """
    Converts a string representation of a base 4 number to its base 10 integer equivalent.

    Args:
      base4_string: The string representing the number in base 4.
                    Should only contain characters '0', '1', '2', '3'.

    Returns:
      The integer (base 10) equivalent of the input base 4 string.
    """
    if not base4_string:
        raise ValueError("Input string cannot be empty.")

    # Check for invalid characters (optional but good practice)
    valid_chars = set("0123")
    if not set(base4_string).issubset(valid_chars):
        raise ValueError(
            f"Invalid character found in base 4 string: '{base4_string}'. Only '0', '1', '2', '3' are allowed."
        )

    return int(base4_string, 4)


def _load_and_pair_files(data_dir: str) -> list[tuple[str, str]]:
    """Find paired audio files and StepMania chart files."""
    return pairing.list_audio_chart_pairs(data_dir)


def resolve_dense_eval_samples(
    dataset_config: config.OnsetDatasetConfig,
    *,
    data_ref: str = "",
    split: str | None = "val",
) -> tuple[list[tuple[str, str, int]], str]:
    """Resolve validation samples and data root for dense event-F1 eval.

    Uses ``training_index.json`` when ``dataset_config.training_index_path`` is
    set or when ``data_ref`` points at a manifest; otherwise falls back to legacy
    ``.txt`` pairs under ``val_data_dir``.

    Args:
        dataset_config: Dense onset dataset configuration.
        data_ref: Optional override for manifest or val directory path.
        split: Manifest split filter (default ``val``).

    Returns:
        ``(samples, data_root)`` where each sample is
        ``(audio_path, chart_path, chart_index)``.
    """
    index_ref = str(dataset_config.training_index_path).strip()
    ref = data_ref or index_ref or dataset_config.val_data_dir
    index_path, data_root = training_index.locate_training_index(ref)
    if index_path is not None or index_ref:
        manifest_ref = str(index_path) if index_path is not None else index_ref
        if index_path is not None:
            index = training_index.load_training_index(index_path)
            resolved_root = str(training_index.resolve_output_dir(index, index_path))
        else:
            resolved_root = str(pathlib.Path(index_ref).resolve().parent)
        configured_root = str(dataset_config.data_root).strip()
        root = configured_root or resolved_root
        samples = list_dense_onset_samples(manifest_ref, split=split)
        if not samples:
            raise ValueError(f"no dense eval samples for split={split!r} under {ref!r}")
        return samples, root

    pairs = pairing.list_audio_chart_pairs(ref)
    if not pairs:
        raise ValueError(f"no audio-chart pairs found under {ref!r}")
    return [(audio_path, chart_path, 0) for audio_path, chart_path in pairs], ref


def dense_eval_sample_key(
    audio_path: str,
    chart_path: str,
    chart_index: int,
) -> str:
    """Return a stable report key for one dense eval sample."""
    del audio_path
    chart = pathlib.Path(chart_path)
    if chart_index:
        return f"{chart.stem}#{chart_index}"
    return chart.parent.name


def list_dense_onset_samples(
    data_dir: str,
    split: str | None = None,
) -> list[tuple[str, str, int]]:
    """Return ``(audio_path, chart_path, chart_index)`` refs for dense onset training.

    Uses ``training_index.json`` / ``.chart.json`` when available; otherwise legacy
    ``.txt`` pairs with ``chart_index`` 0.

    Args:
        data_dir: Manifest file, prepared output root, or legacy training directory.
        split: Optional ``train`` or ``val`` filter when loading from a manifest.

    Returns:
        Sorted sample refs for the dense onset dataloader.
    """
    samples = list_training_samples(data_dir, split=split)
    if samples:
        return samples
    return [
        (audio_path, chart_path, 0)
        for audio_path, chart_path in pairing.list_audio_chart_pairs(data_dir)
    ]


def select_song_pairs(
    pairs: list[Any],
    max_songs: int = -1,
    seed: int | None = None,
) -> list[Any]:
    """Select a reproducible subset of audio/chart pairs for training.

    Pairs are sorted by audio path, shuffled with ``seed``, then truncated to
    ``max_songs``.

    Args:
        pairs: Full list of ``(audio_path, chart_path, ...)`` tuples.
        max_songs: Maximum songs to keep (-1 for all).
        seed: Random seed for shuffling before truncation.

    Returns:
        Selected pairs (all pairs when ``max_songs`` is -1 or exceeds pair count).
    """
    if max_songs == -1 or len(pairs) <= max_songs:
        return list(pairs)
    ordered = sorted(pairs, key=lambda pair: pair[0])
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(ordered))
    return [ordered[index] for index in shuffled_indices[:max_songs]]


def list_audio_chart_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Return paired audio and chart file paths found under a data directory.

    Args:
        data_dir: Root directory to search recursively.

    Returns:
        List of ``(audio_path, chart_path)`` tuples with matching filename stems.
    """
    return pairing.list_audio_chart_pairs(data_dir)


def list_training_samples(
    data_dir: str,
    split: str | None = None,
) -> list[tuple[str, str, int]]:
    """Return ``(audio_path, chart_path, chart_index)`` training sample refs."""
    return pairing.list_training_samples(data_dir, split=split)  # type: ignore[arg-type]


def _is_chart_json(chart_path: str | pathlib.Path) -> bool:
    return pathlib.Path(chart_path).suffix.lower() == ".json"


def _parse_step_chart_impl(
    chart_path: str,
    binary_timings: bool,
    return_bpm: bool,
    *,
    chart_index: int = 0,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float]:
    """Parse StepMania chart data; return times, cols, and optionally BPM."""
    if _is_chart_json(chart_path):
        times_arr = training_loader.load_chart_times_sec(chart_path, chart_index)
        cols_arr = training_loader.load_chart_column_codes(
            chart_path,
            chart_index,
            binary_timings=binary_timings,
        )
        if return_bpm:
            return times_arr, cols_arr, training_loader.load_chart_bpm(chart_path)
        return times_arr, cols_arr

    with pathlib.Path(chart_path).open() as f:
        f.readline()  # TITLE
        bpm_line = f.readline()
        bpm = float(bpm_line.removeprefix("BPM").strip())
        f.readline()  # NOTES
        difficulty_level = f.readline().strip().lower().split(" ")[1]
        _ = _DIFFICULTY_MAP.get(difficulty_level, 2)
        times = []
        cols = []
        for line in f:
            if line.startswith("DIFFICULTY"):
                break
            arrows, timing = line.strip().split(" ")
            times.append(float(timing))
            if binary_timings:
                cols.append(0)
            else:
                cols.append(_base4_to_int(arrows))
    times_arr = np.array(times)
    cols_arr = np.array(cols, dtype=np.int32)
    if return_bpm:
        return times_arr, cols_arr, bpm
    return times_arr, cols_arr


def _parse_step_chart(
    chart_path: str,
    binary_timings: bool = False,
    *,
    chart_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse StepMania chart data to extract step timings and note encodings.

    Args:
        chart_path: Path to legacy ``.txt`` / ``.sm`` or prepared ``.chart.json``.
        binary_timings: If True, returns 0 for all note encodings, effectively
                        treating the output as binary (step vs. no step).
        chart_index: Block index when ``chart_path`` is ``.chart.json``.

    Returns:
        A tuple containing an array of step timings and an array of note encodings.
    """
    out = _parse_step_chart_impl(
        chart_path,
        binary_timings,
        return_bpm=False,
        chart_index=chart_index,
    )
    return out[0], out[1]


def _parse_step_chart_with_bpm(
    chart_path: str,
    binary_timings: bool = False,
    *,
    chart_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse StepMania chart data; return step times, note encodings, and BPM.

    Used when BPM is needed (e.g. beat_phase). Same format as _parse_step_chart
    plus BPM in beats per minute.

    Args:
        chart_path: Path to legacy ``.txt`` / ``.sm`` or prepared ``.chart.json``.
        binary_timings: If True, returns 0 for all note encodings.
        chart_index: Block index when ``chart_path`` is ``.chart.json``.

    Returns:
        (times, cols, bpm): times and cols as in _parse_step_chart; bpm as float.
    """
    result = cast(
        tuple[np.ndarray, np.ndarray, float],
        _parse_step_chart_impl(
            chart_path,
            binary_timings,
            return_bpm=True,
            chart_index=chart_index,
        ),
    )
    return result[0], result[1], result[2]


def normalized_intervals_from_times(times_seconds: np.ndarray) -> np.ndarray:
    """Compute inter-step intervals from onset times, normalized to [0, 1] by max interval.

    Used at inference to feed interval_input when the arrow model was trained with use_interval.
    Step 0 gets 0.0 (same as in training). Intervals are time since previous step;
    normalized by the max interval so output is in [0, 1].

    Args:
        times_seconds: (n_steps,) onset times in seconds.

    Returns:
        intervals_norm: (n_steps,) float32 in [0, 1].
    """
    times = np.asarray(times_seconds, dtype=np.float64)
    if len(times) == 0:
        return times.astype(np.float32)
    diffs = np.diff(times)
    intervals = np.concatenate([[0.0], diffs])
    max_iv = float(np.max(intervals)) + 1e-9
    return (intervals / max_iv).astype(np.float32)


def log_normalized_intervals_from_times(times_seconds: np.ndarray) -> np.ndarray:
    """Compute log(1 + interval) from onset times, normalized to [0, 1] by max.

    Used for interval_encoding \"log\" or \"multi\". Step 0 gets 0.0.

    Args:
        times_seconds: (n_steps,) onset times in seconds.

    Returns:
        (n_steps,) float32 in [0, 1].
    """
    times = np.asarray(times_seconds, dtype=np.float64)
    if len(times) == 0:
        return times.astype(np.float32)
    diffs = np.diff(times)
    intervals = np.concatenate([[0.0], diffs])
    log_iv = np.log1p(intervals)
    max_log = float(np.max(log_iv)) + 1e-9
    return (log_iv / max_log).astype(np.float32)


def next_interval_normalized_from_times(times_seconds: np.ndarray) -> np.ndarray:
    """Compute time-to-next-step per position, normalized to [0, 1]; last step 0.

    Used for interval_encoding \"multi\". At step i, value is (times[i+1]-times[i])
    normalized; last step is 0.

    Args:
        times_seconds: (n_steps,) onset times in seconds.

    Returns:
        (n_steps,) float32 in [0, 1].
    """
    times = np.asarray(times_seconds, dtype=np.float64)
    if len(times) == 0:
        return times.astype(np.float32)
    if len(times) == 1:
        return np.array([0.0], dtype=np.float32)
    next_iv = np.diff(times)  # length n_steps - 1
    next_iv = np.concatenate([next_iv, [0.0]])
    max_iv = float(np.max(next_iv)) + 1e-9
    return (next_iv / max_iv).astype(np.float32)


def step_index_normalized(n_steps: int) -> np.ndarray:
    """Step index 0..N-1 normalized to [0, 1] (0 at first step, 1 at last).

    Args:
        n_steps: Number of steps.

    Returns:
        (n_steps,) float32 in [0, 1].
    """
    if n_steps == 0:
        return np.array([], dtype=np.float32)
    if n_steps == 1:
        return np.array([0.0], dtype=np.float32)
    idx = np.arange(n_steps, dtype=np.float64) / (n_steps - 1)
    return idx.astype(np.float32)


def beat_phase_from_times_bpm(times_seconds: np.ndarray, bpm: float) -> np.ndarray:
    """Compute beat phase (time mod beat_duration) / beat_duration per step.

    Phase in [0, 1). Requires bpm > 0. BPM is always read from the chart txt file.

    Args:
        times_seconds: (n_steps,) onset times in seconds.
        bpm: Beats per minute (from chart or elsewhere).

    Returns:
        (n_steps,) float32 in [0, 1).
    """
    times = np.asarray(times_seconds, dtype=np.float64)
    if len(times) == 0 or bpm <= 0:
        return np.zeros_like(times, dtype=np.float32)
    beat_duration = 60.0 / bpm
    phase = np.fmod(times, beat_duration) / beat_duration
    return phase.astype(np.float32)


def aux_interval_target_from_times(times_seconds: np.ndarray) -> np.ndarray:
    """Next-step interval (shift by -1) normalized to [0, 1]; last step 0.

    Target for auxiliary next-interval regression. At step i, value is the
    interval that follows (to step i+1); last step is 0 (masked in loss).

    Args:
        times_seconds: (n_steps,) onset times in seconds.

    Returns:
        (n_steps,) float32 in [0, 1].
    """
    return next_interval_normalized_from_times(times_seconds)


def extract_snippets_from_spec(
    spec: np.ndarray,
    times_seconds: np.ndarray,
    half_frames: int,
) -> np.ndarray:
    """Extract mel snippets around each time from an existing (time_steps, n_mels) spec.

    Used at inference when the arrow model expects snippet input.

    Args:
        spec: Mel spectrogram (time_steps, n_mels), e.g. normalized_spec from generator.
        times_seconds: (n_times,) onset times in seconds.
        half_frames: Half-window in frames (total = 2*half_frames+1).

    Returns:
        snippets: (n_times, n_frames, n_mels) float32.
    """
    n_frames_total = spec.shape[0]
    n_mels = spec.shape[1]
    n_frames_window = 2 * half_frames + 1
    n_times = len(times_seconds)
    snippets = np.zeros((n_times, n_frames_window, n_mels), dtype=np.float32)
    for i, t in enumerate(times_seconds):
        frame_idx = int(round(t / HOP_COEFF))
        start = frame_idx - half_frames
        end = frame_idx + half_frames + 1
        src_start = max(0, start)
        src_end = min(n_frames_total, end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        snippets[i, dst_start:dst_end, :] = spec[src_start:src_end, :]
    return snippets


def audio_to_spectrogram(audio_path: str) -> np.ndarray:
    """Convert an audio file to a mel-spectrogram using LibROSA.

    Args:
        audio_path: Path to the audio file (mp3, ogg, or wav).

    Returns:
        A 2D numpy array representing the mel-spectrogram in decibels,
        with shape (n_mels, time_steps).
    """
    return mel_onset.audio_to_spectrogram(audio_path)


def _load_mono_waveform(audio_path: str) -> np.ndarray:
    """Load peak-normalized mono audio at the onset model sample rate."""
    y, sr = librosa.load(audio_path, sr=_TARGET_SR, mono=True)
    if sr != _TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_TARGET_SR)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return np.asarray(y, dtype=np.float32)


def load_onset_waveform(audio_path: str) -> np.ndarray:
    """Load a waveform trimmed/padded to the dense-onset mel frame grid.

    The returned length is ``onset_frame_count(audio_path) * WAVEFORM_SAMPLES_PER_FRAME``
    so a strided Conv1D frontend with that hop produces one embedding per target frame.

    Args:
        audio_path: Path to the audio file.

    Returns:
        One-dimensional float32 waveform.
    """
    y = _load_mono_waveform(audio_path)
    n_frames = onset_frame_count(audio_path)
    hop = constants.WAVEFORM_SAMPLES_PER_FRAME
    total = n_frames * hop
    y = np.pad(y, (0, total - y.size)) if y.size < total else y[:total]
    return y.astype(np.float32)


def onset_frame_count(audio_path: str) -> int:
    """Return the onset model time-step count for an audio file (mel STFT grid).

    Args:
        audio_path: Path to the audio file.

    Returns:
        Number of time steps along the onset detection grid for this file.
    """
    return mel_onset.onset_frame_count(audio_path)


def load_onset_features(
    audio_path: str,
    feature_source: config.FeatureSource,
    mert_features_dir: str = "",
    data_root: str = "",
) -> np.ndarray:
    """Load onset model input features for one audio file.

    Args:
        audio_path: Path to the audio file.
        feature_source: FeatureSource.MEL, MERT, or WAVEFORM.
        mert_features_dir: Directory of precomputed ``.mert.npy`` files (MERT only).
        data_root: Training data root for nested MERT paths (MERT only).

    Returns:
        Feature array with shape ``(time_steps, feature_dim)`` for mel/MERT, or
        one-dimensional waveform for WAVEFORM.
    """
    if feature_source == config.FeatureSource.WAVEFORM:
        return load_onset_waveform(audio_path)
    if feature_source == config.FeatureSource.MERT:
        features = ssl_features.load_mert_features(
            audio_path,
            mert_features_dir,
            data_root,
        )
        n_frames = onset_frame_count(audio_path)
        if features.shape[0] != n_frames:
            features = ssl_features.resample_features_to_frame_count(features, n_frames)
        return features
    spec = audio_to_spectrogram(audio_path)
    return np.transpose(spec).astype(np.float32)


def _temporal_augment_scipy(
    spec: np.ndarray | torch.Tensor,
    labels_and_features: np.ndarray | torch.Tensor,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random time warping augmentation for spec, labels, and extra features."""
    spec = (
        spec.detach().cpu().numpy()
        if isinstance(spec, torch.Tensor)
        else np.asarray(spec)
    )
    labels_and_features = (
        labels_and_features.detach().cpu().numpy()
        if isinstance(labels_and_features, torch.Tensor)
        else np.asarray(labels_and_features)
    )

    original_length = spec.shape[1]
    num_extra_channels = labels_and_features.shape[1]

    warp_factor = np.random.uniform(0.85, 1.15)
    new_length = int(original_length * warp_factor)

    spec_resized = np.zeros((n_features, new_length), dtype=spec.dtype)
    original_time = np.arange(original_length)
    warped_time = np.linspace(0, original_length - 1, new_length)

    for bin_idx in range(n_features):
        interp_func = interpolate.interp1d(
            original_time,
            spec[bin_idx, :],
            kind="linear",
            fill_value="extrapolate",  # type: ignore
        )
        spec_resized[bin_idx, :] = interp_func(warped_time)

    extras_resized = np.zeros(
        (new_length, num_extra_channels), dtype=labels_and_features.dtype
    )
    for target_bin in range(num_extra_channels):
        interp_func_labels = interpolate.interp1d(
            original_time,
            labels_and_features[:, target_bin],
            kind="nearest",
            fill_value="extrapolate",  # type: ignore
        )
        extras_resized[:, target_bin] = interp_func_labels(warped_time)

    if new_length > original_length:
        spec = spec_resized[:, :original_length]
        labels_and_features = extras_resized[:original_length, :]
    else:
        pad_width = original_length - new_length
        spec = np.pad(spec_resized, ((0, 0), (0, pad_width)), mode="edge")
        labels_and_features = np.pad(
            extras_resized, ((0, pad_width), (0, 0)), mode="constant"
        )

    return spec, labels_and_features


def _apply_spec_augment(
    spec: np.ndarray,
    max_freq_mask: int = 27,
    max_time_mask: int = 50,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
) -> np.ndarray:
    """
    Applies SpecAugment to a spectrogram.

    Args:
        spec: The input spectrogram of shape (n_mels, time_steps).
        max_freq_mask: The maximum width of the frequency mask.
        max_time_mask: The maximum width of the time mask.
        num_freq_masks: The number of frequency masks to apply.
        num_time_masks: The number of time masks to apply.

    Returns:
        The augmented spectrogram.
    """
    spec_augmented = spec.copy()
    n_freq_bins, time_steps = spec.shape

    # Apply frequency masking
    freq_cap = min(max_freq_mask, n_freq_bins)
    if freq_cap > 0:
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_cap)
            f0 = np.random.randint(0, max(1, n_freq_bins - f))
            spec_augmented[f0 : f0 + f, :] = 0

    # Apply time masking
    time_cap = min(max_time_mask, time_steps)
    if time_cap > 0:
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_cap)
            t0 = np.random.randint(0, max(1, time_steps - t))
            spec_augmented[:, t0 : t0 + t] = 0

    return spec_augmented


def _onset_frame_indices(times: np.ndarray) -> np.ndarray:
    """Snap onset times to their nearest frame on the hop grid.

    Rounding rather than truncating matters because peak-picking reports a
    predicted time of ``frame * HOP_COEFF``: truncation would bias every target
    early by up to a full hop (measured ~4.2 ms mean on ladder charts, a fifth
    of the 20 ms match tolerance), while rounding is unbiased and halves the
    worst-case error.

    Args:
        times: Onset times in seconds.

    Returns:
        Integer frame indices on the ``HOP_COEFF`` grid.
    """
    return np.rint(np.asarray(times, dtype=np.float64) / HOP_COEFF).astype(int)


def _create_target(times: np.ndarray, cols: np.ndarray, spec_length: int) -> np.ndarray:
    """Create target vector from step times and columns."""
    target = np.zeros((spec_length, _N_TARGET), dtype=np.float32)
    for frame_idx, col in zip(_onset_frame_indices(times), cols, strict=False):
        if frame_idx < spec_length:
            target[frame_idx, col] = 1.0
    return target


def _create_target_gaussian(
    times: np.ndarray, cols: np.ndarray, spec_length: int, sigma: float = 1.5
) -> np.ndarray:
    """
    Create target vector with Gaussian distributions around onset times.
    This encourages the model to predict onsets near the ground truth, not just exactly on it.
    """
    target = np.zeros((spec_length, _N_TARGET), dtype=np.float32)

    if times.size == 0:
        return target

    frame_indices = _onset_frame_indices(times)

    kernel_width = int(3 * sigma)
    x = np.arange(-kernel_width, kernel_width + 1)
    gaussian_kernel = np.exp(-(x**2) / (2 * sigma**2))

    for frame_idx, col in zip(frame_indices, cols, strict=False):
        if col >= _N_TARGET:
            continue

        start = max(0, frame_idx - kernel_width)
        end = min(spec_length, frame_idx + kernel_width + 1)

        kernel_start = start - (frame_idx - kernel_width)
        kernel_end = end - (frame_idx - kernel_width)

        target[start:end, col] = np.maximum(
            target[start:end, col], gaussian_kernel[kernel_start:kernel_end]
        )
    return target


def onset_density_scalar(n_onsets: int, duration_sec: float) -> float:
    """Map an onset count and duration to a unit density conditioning value.

    Matches ``onset_ar.config.compute_density_scalar`` in ``onset_density`` mode
    so both tracks condition on the same scale. The AR track chose measured
    density over the simfile difficulty label because those labels are partly
    wrong (NOTE-20260803-01).

    Args:
        n_onsets: Ground-truth onset count for the chart.
        duration_sec: Audio duration in seconds.

    Returns:
        Density scalar clipped to ``[0, 1]``.
    """
    if duration_sec <= 0.0:
        return 0.0
    hz = float(n_onsets) / float(duration_sec)
    return float(np.clip(hz / config.DENSITY_ONSET_HZ_NORM, 0.0, 1.0))


def append_density_channel(features: np.ndarray, density: float) -> np.ndarray:
    """Append a constant density channel to frame features.

    Appended after normalization on purpose: per-bin normalization divides by
    the standard deviation across time, which is zero for a constant channel
    and would erase the conditioning signal.

    Args:
        features: Frame features of shape ``(T, n_features)``.
        density: Density scalar broadcast across every frame.

    Returns:
        Features of shape ``(T, n_features + 1)``.
    """
    channel = np.full((features.shape[0], 1), density, dtype=np.float32)
    return np.concatenate([features.astype(np.float32), channel], axis=1)


def _load_and_preprocess_paths(
    audio_path: str,
    chart_path: str,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    feature_source: config.FeatureSource,
    mert_features_dir: str,
    data_root: str,
    *,
    chart_index: int = 0,
    density_conditioning: str = config.DENSITY_CONDITIONING_NONE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load audio and chart, build features and target (pure Python, no TF)."""
    features = load_onset_features(
        audio_path,
        feature_source,
        mert_features_dir,
        data_root,
    )
    if feature_source == config.FeatureSource.WAVEFORM:
        spec_length = features.size // constants.WAVEFORM_SAMPLES_PER_FRAME
    else:
        # Normalize here, not in the augment map: the augment map is skipped
        # when no augmentation is enabled, and inference normalizes too, so
        # doing it there would train on a different scale than eval scores.
        features = normalize_onset_spectrogram(features)
        spec_length = features.shape[0]
    times, cols = _parse_step_chart(
        chart_path,
        binary_timings=True,
        chart_index=chart_index,
    )
    target = (
        _create_target_gaussian(times, cols, spec_length, gaussian_sigma)
        if use_gaussian_target
        else _create_target(times, cols, spec_length)
    )
    if density_conditioning != config.DENSITY_CONDITIONING_NONE:
        features = append_density_channel(
            features,
            onset_density_scalar(int(np.size(times)), spec_length * HOP_COEFF),
        )
    return features.astype(np.float32), target.astype(np.float32)


def _augment_features_numpy(
    features: np.ndarray,
    target: np.ndarray,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply optional temporal/spec augmentation (pure Python).

    Normalization happens in the load path so it applies whether or not
    augmentation runs; see :func:`_load_and_preprocess_paths`.
    """
    if features.ndim == 1:
        return features.astype(np.float32), target.astype(np.float32)
    spec_py = np.transpose(features[:, :n_features])
    combined_labels = target
    if apply_temporal_augment:
        spec_py, combined_labels = _temporal_augment_scipy(
            spec_py, combined_labels, n_features
        )
    if should_apply_spec_augment:
        spec_py = _apply_spec_augment(spec_py, max_freq_mask=int(0.2 * n_features))
    final_target = combined_labels[:, :_N_TARGET]
    final_features = np.transpose(spec_py)
    return final_features.astype(np.float32), final_target.astype(np.float32)


def _load_arrow_pair_numpy(
    audio_path: str,
    chart_path: str,
    snippet_half_frames: int,
    use_interval: bool,
    interval_encoding: config.IntervalEncoding,
    use_step_index: bool,
    use_beat_phase: bool,
    use_aux_interval_target: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load and featurize one ``(audio_path, chart_path)`` pair for arrow training."""
    if use_beat_phase:
        times, cols, bpm = _parse_step_chart_with_bpm(chart_path, binary_timings=False)
    else:
        times, cols = _parse_step_chart(chart_path, binary_timings=False)
        bpm = 120.0

    times = np.asarray(times, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.int32)
    n_steps = len(times)
    n_frames = (2 * snippet_half_frames + 1) if snippet_half_frames > 0 else 0
    zeros_n = np.zeros(n_steps, dtype=np.float32)

    if n_steps == 0:
        return (
            times.astype(np.float32),
            times.astype(np.float32),
            zeros_n,
            np.zeros((0, n_frames, _N_MELS), dtype=np.float32),
            cols,
            zeros_n,
            zeros_n,
            zeros_n,
            np.array([], dtype=np.float32),
        )

    times_norm = (times / (np.max(times) + 1e-9)).astype(np.float32)
    if use_interval:
        if interval_encoding == config.IntervalEncoding.LOG:
            intervals_norm = log_normalized_intervals_from_times(times)
            interval_log_norm = zeros_n.copy()
        elif interval_encoding == config.IntervalEncoding.MULTI:
            intervals_norm = next_interval_normalized_from_times(times)
            interval_log_norm = log_normalized_intervals_from_times(times)
        elif interval_encoding == config.IntervalEncoding.DEFAULT:
            intervals_norm = normalized_intervals_from_times(times)
            interval_log_norm = zeros_n.copy()
        else:
            raise ValueError(f"Invalid interval encoding: {interval_encoding}")
    else:
        intervals_norm = zeros_n.copy()
        interval_log_norm = zeros_n.copy()

    step_index = step_index_normalized(n_steps) if use_step_index else zeros_n.copy()

    if use_beat_phase and bpm > 0:
        beat_phase = beat_phase_from_times_bpm(times, bpm)
    else:
        beat_phase = zeros_n.copy()

    if use_aux_interval_target:
        aux_interval = aux_interval_target_from_times(times)
        aux_interval_mask = np.ones(n_steps, dtype=np.float32)
        aux_interval_mask[-1] = 0.0
    else:
        aux_interval = zeros_n.copy()
        aux_interval_mask = zeros_n.copy()

    if snippet_half_frames > 0:
        spec_time_major = normalize_onset_spectrogram(
            audio_to_spectrogram(audio_path).T
        )
        snippets = extract_snippets_from_spec(
            spec_time_major, times, snippet_half_frames
        )
    else:
        snippets = np.zeros((n_steps, 0, _N_MELS), dtype=np.float32)

    return (
        times_norm,
        intervals_norm,
        interval_log_norm,
        snippets,
        cols,
        step_index,
        beat_phase,
        aux_interval,
        aux_interval_mask,
    )


def _arrow_use_dict_output(
    snippet_half_frames: int,
    use_interval: bool,
    use_step_index: bool,
    use_beat_phase: bool,
    use_aux_interval_target: bool,
) -> bool:
    """Return True if arrow dataset should yield ``(dict, cols)``; else ``(times, cols)``."""
    return (
        snippet_half_frames > 0
        or use_interval
        or use_step_index
        or use_beat_phase
        or use_aux_interval_target
    )


_ORDER_EPS = 1e-6


def _apply_timing_jitter_py_callback(
    features: dict[str, np.ndarray] | np.ndarray,
    cols: np.ndarray,
    sigma: float,
    use_dict: bool,
    use_interval: bool,
    interval_encoding: config.IntervalEncoding,
) -> tuple[dict[str, np.ndarray] | np.ndarray, np.ndarray]:
    """Apply Gaussian jitter to ``timing_input`` and recompute intervals.

    Args:
        features: Dict of ``(n_steps, 1)`` arrays or single ``(n_steps, 1)`` array.
        cols: ``(n_steps,)`` column labels, returned unchanged.
        sigma: Gaussian std for jitter in ``[0, 1]``.
        use_dict: True if ``features`` is a dict.
        use_interval: True if interval inputs should be recomputed.
        interval_encoding: How intervals are encoded (DEFAULT, LOG, MULTI).

    Returns:
        ``(features_jittered, cols)`` with the same structure as input.

    Raises:
        ValueError: If interval encoding is invalid.
    """
    if sigma <= 0:
        return features, cols

    def _jitter_and_clip(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float64).flatten()
        noise = np.random.default_rng().normal(0, sigma, size=out.shape)
        out = np.clip(out + noise, 0.0, 1.0)
        return out.astype(np.float32).reshape(arr.shape)

    def _enforce_order(times: np.ndarray) -> np.ndarray:
        t = times.astype(np.float64).flatten()
        for i in range(1, len(t)):
            t[i] = max(t[i], t[i - 1] + _ORDER_EPS)
        return np.clip(t, 0.0, 1.0).astype(np.float32).reshape(times.shape)

    if not use_dict:
        times = np.asarray(features, dtype=np.float32)
        jittered = _enforce_order(_jitter_and_clip(times))
        return jittered, cols

    out = dict(features)
    timing = np.asarray(out["timing_input"], dtype=np.float32)
    jittered_timing = _enforce_order(_jitter_and_clip(timing))
    out["timing_input"] = jittered_timing

    has_interval = (
        use_interval
        or "interval_input" in out
        or "interval_log_input" in out
        or "interval_next_input" in out
    )
    if has_interval:
        t_flat = jittered_timing.flatten()
        if interval_encoding == config.IntervalEncoding.DEFAULT:
            intervals = normalized_intervals_from_times(t_flat)
            out["interval_input"] = np.expand_dims(intervals, axis=-1)
        elif interval_encoding == config.IntervalEncoding.LOG:
            intervals = log_normalized_intervals_from_times(t_flat)
            out["interval_log_input"] = np.expand_dims(intervals, axis=-1)
        elif interval_encoding == config.IntervalEncoding.MULTI:
            interval_log = log_normalized_intervals_from_times(t_flat)
            interval_next = next_interval_normalized_from_times(t_flat)
            out["interval_log_input"] = np.expand_dims(interval_log, axis=-1)
            out["interval_next_input"] = np.expand_dims(interval_next, axis=-1)
        else:
            raise ValueError(f"Invalid interval encoding: {interval_encoding}")

    return out, cols


def _random_crop_frames_numpy(
    features: np.ndarray,
    target: np.ndarray,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a random fixed-length window from one ``(features, target)`` song pair."""
    n_frames = features.shape[0]
    if n_frames < window_frames:
        pad = window_frames - n_frames
        features = np.pad(features, ((0, pad), (0, 0)), mode="constant")
        target = np.pad(target, ((0, pad), (0, 0)), mode="constant")
        n_frames = window_frames
    max_start = n_frames - window_frames
    start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
    return (
        features[start : start + window_frames].astype(np.float32),
        target[start : start + window_frames].astype(np.float32),
    )


class DenseOnsetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for dense onset detection with in-memory feature caching."""

    def __init__(
        self,
        samples: list[tuple[str, str, int]],
        *,
        use_gaussian_target: bool = False,
        gaussian_sigma: float = 1.0,
        feature_source: config.FeatureSource = config.FeatureSource.MEL,
        mert_features_dir: str = "",
        data_root: str = "",
        n_features: int = _N_MELS,
        apply_temporal_augment: bool = False,
        should_apply_spec_augment: bool = False,
        window_frames: int = 0,
        windows_per_song: int = 1,
        density_conditioning: str = config.DENSITY_CONDITIONING_NONE,
    ) -> None:
        self.samples = list(samples)
        self.use_gaussian_target = use_gaussian_target
        self.gaussian_sigma = gaussian_sigma
        self.feature_source = feature_source
        self.mert_features_dir = mert_features_dir
        self.data_root = data_root
        self.n_features = n_features
        self.apply_temporal_augment = apply_temporal_augment
        self.should_apply_spec_augment = should_apply_spec_augment
        self.window_frames = int(window_frames)
        self.windows_per_song = max(1, int(windows_per_song))
        self.density_conditioning = density_conditioning
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        mult = self.windows_per_song if self.window_frames > 0 else 1
        return len(self.samples) * mult

    def _get_cached(self, song_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if song_idx not in self._cache:
            audio_path, chart_path, chart_index = self.samples[song_idx]
            self._cache[song_idx] = _load_and_preprocess_paths(
                audio_path,
                chart_path,
                self.use_gaussian_target,
                self.gaussian_sigma,
                self.feature_source,
                self.mert_features_dir,
                self.data_root,
                chart_index=chart_index,
                density_conditioning=self.density_conditioning,
            )
        return self._cache[song_idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        song_idx = index % len(self.samples)
        features, target = self._get_cached(song_idx)
        if self.window_frames > 0:
            features, target = _random_crop_frames_numpy(
                features, target, self.window_frames
            )
        if self.apply_temporal_augment or self.should_apply_spec_augment:
            features, target = _augment_features_numpy(
                features,
                target,
                self.apply_temporal_augment,
                self.should_apply_spec_augment,
                self.n_features,
            )
        return (
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)),
        )


def _collate_dense_onset(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 1:
        x, y = batch[0]
        return x.unsqueeze(0), y.unsqueeze(0)
    feats = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_t = max(t.shape[0] for t in targets)
    max_x = max(x.shape[0] for x in feats)
    if feats[0].ndim == 1:
        x_out = torch.zeros((len(batch), max_x), dtype=torch.float32)
        for i, x in enumerate(feats):
            x_out[i, : x.shape[0]] = x
    else:
        feat_dim = feats[0].shape[1]
        x_out = torch.zeros((len(batch), max_x, feat_dim), dtype=torch.float32)
        for i, x in enumerate(feats):
            x_out[i, : x.shape[0]] = x
    y_out = torch.zeros((len(batch), max_t, targets[0].shape[-1]), dtype=torch.float32)
    for i, y in enumerate(targets):
        y_out[i, : y.shape[0]] = y
    return x_out, y_out


class ArrowSequenceDataset(Dataset[tuple[Any, torch.Tensor]]):
    """PyTorch dataset for arrow classification with in-memory caching and jitter."""

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        *,
        snippet_half_frames: int = 0,
        use_interval: bool = False,
        interval_encoding: config.IntervalEncoding = config.IntervalEncoding.DEFAULT,
        use_step_index: bool = False,
        use_beat_phase: bool = False,
        use_aux_interval_target: bool = False,
        timing_jitter_sigma: float = 0.0,
    ) -> None:
        self.pairs = list(pairs)
        self.snippet_half_frames = int(snippet_half_frames)
        self.use_interval = bool(use_interval)
        self.interval_encoding = interval_encoding
        self.use_step_index = bool(use_step_index)
        self.use_beat_phase = bool(use_beat_phase)
        self.use_aux_interval_target = bool(use_aux_interval_target)
        self.timing_jitter_sigma = float(timing_jitter_sigma)
        self.use_dict_output = _arrow_use_dict_output(
            self.snippet_half_frames,
            self.use_interval,
            self.use_step_index,
            self.use_beat_phase,
            self.use_aux_interval_target,
        )
        self._cache: dict[
            int, tuple[dict[str, np.ndarray] | np.ndarray, np.ndarray]
        ] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_cached(
        self, index: int
    ) -> tuple[dict[str, np.ndarray] | np.ndarray, np.ndarray]:
        if index not in self._cache:
            audio_path, chart_path = self.pairs[index]
            (
                times,
                intervals,
                interval_log,
                snippets,
                cols,
                step_index,
                beat_phase,
                aux_interval,
                aux_interval_mask,
            ) = _load_arrow_pair_numpy(
                audio_path,
                chart_path,
                self.snippet_half_frames,
                self.use_interval,
                self.interval_encoding,
                self.use_step_index,
                self.use_beat_phase,
                self.use_aux_interval_target,
            )
            times_col = times[:, np.newaxis]
            if self.use_dict_output:
                out: dict[str, np.ndarray] = {"timing_input": times_col}
                if self.use_interval:
                    if self.interval_encoding == config.IntervalEncoding.DEFAULT:
                        out["interval_input"] = intervals[:, np.newaxis]
                    elif self.interval_encoding == config.IntervalEncoding.LOG:
                        out["interval_log_input"] = intervals[:, np.newaxis]
                    elif self.interval_encoding == config.IntervalEncoding.MULTI:
                        out["interval_log_input"] = interval_log[:, np.newaxis]
                        out["interval_next_input"] = intervals[:, np.newaxis]
                if self.use_step_index:
                    out["step_index_input"] = step_index[:, np.newaxis]
                if self.use_beat_phase:
                    out["beat_phase_input"] = beat_phase[:, np.newaxis]
                if self.snippet_half_frames > 0:
                    out["snippet_input"] = snippets
                if self.use_aux_interval_target:
                    out["aux_interval_target"] = aux_interval[:, np.newaxis]
                    out["aux_interval_mask"] = aux_interval_mask[:, np.newaxis]
                self._cache[index] = (out, cols)
            else:
                self._cache[index] = (times_col, cols)
        return self._cache[index]

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor] | torch.Tensor, torch.Tensor]:
        features, cols = self._get_cached(index)
        if self.timing_jitter_sigma > 0:
            features, cols = _apply_timing_jitter_py_callback(
                features,
                cols,
                self.timing_jitter_sigma,
                self.use_dict_output,
                self.use_interval,
                self.interval_encoding,
            )
        cols_t = torch.from_numpy(np.ascontiguousarray(cols, dtype=np.int64))
        if isinstance(features, dict):
            feats_t = {
                k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
                for k, v in features.items()
            }
            return feats_t, cols_t
        return torch.from_numpy(
            np.ascontiguousarray(features, dtype=np.float32)
        ), cols_t


def _collate_arrow_batch(
    batch: list[tuple[dict[str, torch.Tensor] | torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor] | torch.Tensor, torch.Tensor]:
    if len(batch) == 1:
        feats, cols = batch[0]
        if isinstance(feats, dict):
            return {k: v.unsqueeze(0) for k, v in feats.items()}, cols.unsqueeze(0)
        return feats.unsqueeze(0), cols.unsqueeze(0)

    max_len = max(int(cols.shape[0]) for _, cols in batch)
    cols_out = torch.full(
        (len(batch), max_len),
        constants.ARROW_PADDING_CLASS,
        dtype=torch.long,
    )
    for i, (_, cols) in enumerate(batch):
        cols_out[i, : cols.shape[0]] = cols

    first_feats = batch[0][0]
    if isinstance(first_feats, dict):
        dict_out: dict[str, torch.Tensor] = {}
        for key in first_feats:
            tail_shape = first_feats[key].shape[1:]
            padded = torch.zeros(
                (len(batch), max_len, *tail_shape), dtype=torch.float32
            )
            for i, (feats, _) in enumerate(batch):
                assert isinstance(feats, dict)
                t = feats[key]
                padded[i, : t.shape[0]] = t
            dict_out[key] = padded
        return dict_out, cols_out

    padded_t = torch.zeros((len(batch), max_len, 1), dtype=torch.float32)
    for i, (feats, _) in enumerate(batch):
        assert isinstance(feats, torch.Tensor)
        padded_t[i, : feats.shape[0]] = feats
    return padded_t, cols_out


def create_dataset(
    data_dir: str,
    batch_size: int = 1,
    apply_temporal_augment: bool = False,
    should_apply_spec_augment: bool = False,
    use_gaussian_target: bool = False,
    gaussian_sigma: float = 1.0,
    feature_source: config.FeatureSource = config.FeatureSource.MEL,
    mert_features_dir: str = "",
    n_features: int | None = None,
    max_songs: int = -1,
    song_selection_seed: int | None = None,
    split: str | None = None,
    data_root: str = "",
    window_frames: int = 0,
    windows_per_song: int = 1,
    density_conditioning: str = config.DENSITY_CONDITIONING_NONE,
) -> DataLoader[Any]:
    """Create a PyTorch DataLoader for dense onset training or validation.

    Args:
        data_dir: Directory or training manifest path.
        batch_size: Batch size.
        apply_temporal_augment: Apply random time-warping augmentation.
        should_apply_spec_augment: Apply SpecAugment masking.
        use_gaussian_target: Generate Gaussian-smoothed onset targets.
        gaussian_sigma: Gaussian target standard deviation in frames.
        feature_source: Feature source (MEL, MERT, or WAVEFORM).
        mert_features_dir: Directory of precomputed MERT features.
        n_features: Optional input feature dimension override.
        max_songs: Maximum songs to select (``-1`` keeps all).
        song_selection_seed: Optional seed for song subset selection.
        split: Optional manifest split (``train`` or ``val``).
        data_root: Optional data root directory.
        window_frames: Random crop window length (``0`` disables cropping).
        windows_per_song: Random crops drawn per song per epoch.
        density_conditioning: Density conditioning mode.

    Returns:
        PyTorch :class:`DataLoader` yielding ``(features, targets)`` tensor batches.

    Raises:
        ValueError: If no audio-chart pairs are found or ``window_frames`` is used
            with ``FeatureSource.WAVEFORM``.
    """
    if n_features is None:
        if feature_source == config.FeatureSource.MERT:
            n_features = constants.MERT_HIDDEN_SIZE
        else:
            n_features = _N_MELS

    if window_frames > 0 and feature_source == config.FeatureSource.WAVEFORM:
        raise ValueError("window_frames requires a frame feature source (mel/mert)")

    root = data_root or data_dir
    samples = select_song_pairs(
        list_dense_onset_samples(data_dir, split=split),
        max_songs=max_songs,
        seed=song_selection_seed,
    )
    if not samples:
        raise ValueError("No audio-chart pairs found.")

    ds = DenseOnsetDataset(
        samples,
        use_gaussian_target=use_gaussian_target,
        gaussian_sigma=gaussian_sigma,
        feature_source=feature_source,
        mert_features_dir=mert_features_dir,
        data_root=root,
        n_features=n_features,
        apply_temporal_augment=apply_temporal_augment,
        should_apply_spec_augment=should_apply_spec_augment,
        window_frames=window_frames,
        windows_per_song=windows_per_song,
        density_conditioning=density_conditioning,
    )
    shuffle = window_frames > 0
    drop_last = window_frames > 0 and len(ds) >= batch_size
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=_collate_dense_onset,
    )


def create_arrow_dataset(
    data_dir: str,
    batch_size: int = 1,
    snippet_half_frames: int = 0,
    use_interval: bool = False,
    interval_encoding: config.IntervalEncoding = config.IntervalEncoding.DEFAULT,
    use_step_index: bool = False,
    use_beat_phase: bool = False,
    use_aux_interval_target: bool = False,
    timing_jitter_sigma: float = 0.0,
) -> DataLoader[Any]:
    """Create a PyTorch DataLoader for arrow classification.

    Args:
        data_dir: Directory containing audio and chart files.
        batch_size: Number of songs per batch.
        snippet_half_frames: Half-window of mel frames around each onset.
        use_interval: Include inter-step interval features.
        interval_encoding: Interval encoding variant (DEFAULT, LOG, MULTI).
        use_step_index: Include normalized step index input.
        use_beat_phase: Include beat phase input.
        use_aux_interval_target: Include auxiliary next-interval targets and mask.
        timing_jitter_sigma: Gaussian jitter sigma applied to timings on the fly.

    Returns:
        PyTorch :class:`DataLoader` yielding ``(inputs, cols)`` batches.

    Raises:
        ValueError: If no audio-chart pairs are found.
    """
    pairs = _load_and_pair_files(data_dir)
    if not pairs:
        raise ValueError("No audio-chart pairs found in the specified directory.")

    ds = ArrowSequenceDataset(
        pairs,
        snippet_half_frames=snippet_half_frames,
        use_interval=use_interval,
        interval_encoding=interval_encoding,
        use_step_index=use_step_index,
        use_beat_phase=use_beat_phase,
        use_aux_interval_target=use_aux_interval_target,
        timing_jitter_sigma=timing_jitter_sigma,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_arrow_batch,
    )
