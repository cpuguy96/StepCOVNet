"""Data collection and preprocessing for StepCovNet.

This module provides functionality to load audio and StepMania chart files,
process them into spectrograms and target vectors, and create a TensorFlow
dataset for training.
"""

from typing import Any, cast

import librosa
import numpy as np
import tensorflow as tf
from scipy import interpolate

from stepcovnet import config, constants, pairing, ssl_features

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


def list_audio_chart_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Return paired audio and chart file paths found under a data directory.

    Args:
        data_dir: Root directory to search recursively.

    Returns:
        List of ``(audio_path, chart_path)`` tuples with matching filename stems.
    """
    return pairing.list_audio_chart_pairs(data_dir)


def _parse_step_chart_impl(
    chart_path: str, binary_timings: bool, return_bpm: bool
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, float]:
    """Parse StepMania .sm file; return times, cols, and optionally BPM."""
    with open(chart_path) as f:
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
    chart_path: str, binary_timings: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Parse StepMania .sm file to extract step timings and note encodings.

    Args:
        chart_path: Path to the StepMania .sm file.
        binary_timings: If True, returns 0 for all note encodings, effectively
                        treating the output as binary (step vs. no step).

    Returns:
        A tuple containing an array of step timings and an array of note encodings.
    """
    out = _parse_step_chart_impl(chart_path, binary_timings, return_bpm=False)
    return out[0], out[1]


def _parse_step_chart_with_bpm(
    chart_path: str, binary_timings: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """Parse StepMania .sm file; return step times, note encodings, and BPM.

    Used when BPM is needed (e.g. beat_phase). Same format as _parse_step_chart
    plus BPM in beats per minute.

    Args:
        chart_path: Path to the StepMania .sm file.
        binary_timings: If True, returns 0 for all note encodings.

    Returns:
        (times, cols, bpm): times and cols as in _parse_step_chart; bpm as float.
    """
    result = cast(
        tuple[np.ndarray, np.ndarray, float],
        _parse_step_chart_impl(chart_path, binary_timings, return_bpm=True),
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

    y, sr = librosa.load(audio_path, sr=_TARGET_SR)

    # Sample rate conversion (if necessary)
    if sr != _TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_TARGET_SR)

    # Normalize audio data
    y = y / np.max(np.abs(y))

    hop_length = int(round(_TARGET_SR * HOP_COEFF))
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


def load_onset_features(
    audio_path: str,
    feature_source: config.FeatureSource,
    mert_features_dir: str = "",
    data_root: str = "",
) -> np.ndarray:
    """Load onset model input features for one audio file.

    Args:
        audio_path: Path to the audio file.
        feature_source: FeatureSource.MEL or FeatureSource.MERT.
        mert_features_dir: Directory of precomputed ``.mert.npy`` files (MERT only).
        data_root: Training data root for nested MERT paths (MERT only).

    Returns:
        Feature array with shape ``(time_steps, feature_dim)``, float32.
    """
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
    spec: np.ndarray,
    labels_and_features: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random time warping augmentation for spec, labels, and extra features."""
    spec = spec.numpy() if isinstance(spec, tf.Tensor) else spec  # type: ignore
    labels_and_features = (
        labels_and_features.numpy()  # type: ignore
        if isinstance(labels_and_features, tf.Tensor)
        else labels_and_features
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
    F: int = 27,
    T: int = 50,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
) -> np.ndarray:
    """
    Applies SpecAugment to a spectrogram.

    Args:
        spec: The input spectrogram of shape (n_mels, time_steps).
        F: The maximum width of the frequency mask.
        T: The maximum width of the time mask.
        num_freq_masks: The number of frequency masks to apply.
        num_time_masks: The number of time masks to apply.

    Returns:
        The augmented spectrogram.
    """
    spec_augmented = spec.copy()
    n_freq_bins, time_steps = spec.shape

    # Apply frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, F)
        f0 = np.random.randint(0, n_freq_bins - f)
        spec_augmented[f0 : f0 + f, :] = 0

    # Apply time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, T)
        t0 = np.random.randint(0, time_steps - t)
        spec_augmented[:, t0 : t0 + t] = 0

    return spec_augmented


def _create_target(times: np.ndarray, cols: np.ndarray, spec_length: int) -> np.ndarray:
    """Create target vector from step times and columns."""
    target = np.zeros((spec_length, _N_TARGET), dtype=np.float32)
    for time, col in zip(times, cols, strict=False):
        frame_idx = int(time / HOP_COEFF)
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

    frame_indices = (times / HOP_COEFF).astype(int)

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


def _load_and_preprocess_paths(
    audio_path: str,
    chart_path: str,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    feature_source: config.FeatureSource,
    mert_features_dir: str,
    data_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load audio and chart, build features and target (pure Python, no TF)."""
    features = load_onset_features(
        audio_path,
        feature_source,
        mert_features_dir,
        data_root,
    )
    spec_length = features.shape[0]
    times, cols = _parse_step_chart(chart_path, binary_timings=True)
    target = (
        _create_target_gaussian(times, cols, spec_length, gaussian_sigma)
        if use_gaussian_target
        else _create_target(times, cols, spec_length)
    )
    return features.astype(np.float32), target.astype(np.float32)


def _load_and_preprocess_py_callback(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    feature_source: config.FeatureSource,
    mert_features_dir: str,
    data_root: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode paths and delegate to _load_and_preprocess_paths (for tf.py_function)."""
    audio_path = audio_path_t.numpy().decode()  # type: ignore[union-attr]
    chart_path = chart_path_t.numpy().decode()  # type: ignore[union-attr]
    return _load_and_preprocess_paths(
        audio_path,
        chart_path,
        use_gaussian_target,
        gaussian_sigma,
        feature_source,
        mert_features_dir,
        data_root,
    )


def _load_and_preprocess_tf_map(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    use_gaussian_target: bool,
    gaussian_sigma: float,
    feature_source: config.FeatureSource,
    mert_features_dir: str,
    data_root: str,
    n_features: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map one (audio_path, chart_path) to (features, target) tensors."""
    features, target = tf.py_function(  # type: ignore[misc]
        lambda ap, cp: _load_and_preprocess_py_callback(
            ap,
            cp,
            use_gaussian_target,
            gaussian_sigma,
            feature_source,
            mert_features_dir,
            data_root,
        ),
        [audio_path_t, chart_path_t],
        (tf.float32, tf.float32),
    )
    features.set_shape([None, n_features])
    target.set_shape([None, _N_TARGET])
    return features, target


def _augment_features_numpy(
    features: np.ndarray,
    target: np.ndarray,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply optional temporal/spec augmentation and normalize (pure Python)."""
    spec_py = np.transpose(features[:, :n_features])
    combined_labels = target
    if apply_temporal_augment:
        spec_py, combined_labels = _temporal_augment_scipy(
            spec_py, combined_labels, n_features
        )
    spec_py = normalize_onset_spectrogram(spec_py.T).T
    if should_apply_spec_augment:
        spec_py = _apply_spec_augment(spec_py, F=int(0.2 * n_features))
    final_target = combined_labels[:, :_N_TARGET]
    final_features = np.transpose(spec_py)
    return final_features.astype(np.float32), final_target.astype(np.float32)


def _augment_py_callback(
    features_t: tf.Tensor,
    target_t: tf.Tensor,
    temp_aug: bool,
    spec_aug: bool,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tensors to numpy and delegate to _augment_features_numpy."""
    features = features_t.numpy()  # type: ignore[union-attr]
    target = target_t.numpy()  # type: ignore[union-attr]
    return _augment_features_numpy(features, target, temp_aug, spec_aug, n_features)


def _apply_augmentations_tf_map(
    features: tf.Tensor,
    target: tf.Tensor,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
    n_features: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map (features, target) to augmented (features, target) tensors."""
    aug_features, aug_target = tf.py_function(  # type: ignore[misc]
        lambda f, t: _augment_py_callback(
            f, t, apply_temporal_augment, should_apply_spec_augment, n_features
        ),
        [features, target],
        (tf.float32, tf.float32),
    )
    aug_features.set_shape([None, n_features])
    aug_target.set_shape([None, _N_TARGET])
    return aug_features, aug_target


def _load_arrow_pair_py_callback(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
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
    """Decode (audio_path, chart_path) tensors and load arrow data for tf.py_function.

    Parses chart, optionally loads audio and extracts mel snippets when
    snippet_half_frames > 0, and optionally computes normalized step index and
    extra features (step index, beat phase, aux target).
    Returns a fixed 9-tuple for tf.py_function.

    Args:
        audio_path_t: Tensor containing the audio path.
        chart_path_t: Tensor containing the chart path.
        snippet_half_frames: Half-window of frames around each onset (total = 2*snippet_half_frames+1).
        use_interval: If True, include interval_input (time since previous step) in the batch dict.
        interval_encoding: IntervalEncoding (DEFAULT, LOG, or MULTI). Must match model config.
        use_step_index: If True, include step_index_input (normalized position in sequence).
        use_beat_phase: If True, include beat_phase_input (BPM from chart txt).
        use_aux_interval_target: If True, include aux_interval_target (next-step interval) for aux loss.

    Returns:
        times: (n_steps,) float32, normalized to [0, 1].
        intervals: (n_steps,) float32, normalized when use_interval else zeros.
        interval_log: (n_steps,) float32, log-normalized when interval_encoding multi else zeros.
        snippets: (n_steps, n_frames, n_mels) or (n_steps, 0, n_mels).
        cols: (n_steps,) int32.
        step_index: (n_steps,) float32, normalized [0,1] when use_step_index else zeros.
        beat_phase: (n_steps,) float32, phase when use_beat_phase else zeros.
        aux_interval: (n_steps,) float32, next-step interval target when use_aux else zeros.
        aux_interval_mask: (n_steps,) float32, 1.0 for valid steps, 0.0 for last step (for loss masking).
    """
    audio_path = audio_path_t.numpy().decode()  # type: ignore[union-attr]
    chart_path = chart_path_t.numpy().decode()  # type: ignore[union-attr]

    if use_beat_phase:
        times, cols, bpm = _parse_step_chart_with_bpm(chart_path, binary_timings=False)
    else:
        times, cols = _parse_step_chart(chart_path, binary_timings=False)
        bpm = 120.0  # unused when not use_beat_phase

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
            np.array([], dtype=np.float32),  # aux_interval_mask
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

    if use_step_index:
        step_index = step_index_normalized(n_steps)
    else:
        step_index = zeros_n.copy()

    if use_beat_phase and bpm > 0:
        beat_phase = beat_phase_from_times_bpm(times, bpm)
    else:
        beat_phase = zeros_n.copy()

    if use_aux_interval_target:
        aux_interval = aux_interval_target_from_times(times)
        aux_interval_mask = np.ones(n_steps, dtype=np.float32)
        aux_interval_mask[-1] = 0.0  # last step has no next interval
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
    """Return True if arrow dataset should yield (dict of inputs/targets, cols); else (times, cols)."""
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
    """Apply Gaussian jitter to timing_input and recompute intervals from jittered times.

    Used only during training; called from an uncached map so each epoch sees new noise.
    Jittered values are clipped to [0, 1] and timing order is enforced so intervals stay non-negative.

    Args:
        features: Either a dict of (n_steps, 1) arrays or a single (n_steps, 1) times array.
        cols: (n_steps,) column labels, returned unchanged.
        sigma: Gaussian std for jitter in [0, 1]; applied to timing_input only.
        use_dict: True if features is a dict.
        use_interval: True if interval inputs are present and should be recomputed from jittered times.
        interval_encoding: How intervals are encoded (DEFAULT, LOG, MULTI).

    Returns:
        (features_jittered, cols) with same structure as input.

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
        jittered = _jitter_and_clip(times)
        jittered = _enforce_order(jittered)
        return jittered, cols

    out = dict(features)
    timing = np.asarray(out["timing_input"], dtype=np.float32)
    jittered_timing = _jitter_and_clip(timing)
    jittered_timing = _enforce_order(jittered_timing)
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


# Order of optional dict keys for jitter map flatten/unflatten (must match TF map).
_JITTER_OPTIONAL_KEYS = [
    "step_index_input",
    "interval_input",
    "interval_log_input",
    "interval_next_input",
    "beat_phase_input",
    "snippet_input",
    "aux_interval_target",
    "aux_interval_mask",
]


def _apply_timing_jitter_tf_map(
    features: tf.Tensor | dict[str, tf.Tensor],
    cols: tf.Tensor,
    sigma: float,
    use_dict_output: bool,
    use_interval: bool,
    interval_encoding: config.IntervalEncoding,
    n_frames_window: int,
) -> tuple[tf.Tensor | dict[str, tf.Tensor], tf.Tensor]:
    """Apply timing jitter via py_function; used only when sigma > 0 after cache."""
    # Flatten inputs so tf.py_function gets one list of tensors: [timing, cols, ...optional keys].
    empty_01 = tf.constant([], shape=(0, 1), dtype=tf.float32)
    empty_snippet = tf.zeros((0, n_frames_window, _N_MELS), dtype=tf.float32)

    def _default_tf(key: str) -> tf.Tensor:
        return empty_snippet if key == "snippet_input" else empty_01

    timing_t = features["timing_input"] if use_dict_output else features
    flat = [timing_t, cols] + [
        features.get(k, _default_tf(k)) if use_dict_output else _default_tf(k)
        for k in _JITTER_OPTIONAL_KEYS
    ]

    def _py_jitter(
        timing_t: tf.Tensor, cols_t: tf.Tensor, *optional_t: tf.Tensor
    ) -> tuple:
        # arrs[0]=timing, arrs[1]=cols, arrs[2:]=optionals in _JITTER_OPTIONAL_KEYS order.
        empty_01_np = np.array([], dtype=np.float32).reshape(0, 1)
        empty_snippet_np = np.zeros((0, n_frames_window, _N_MELS), dtype=np.float32)

        def _empty(key: str) -> np.ndarray:
            return empty_snippet_np if key == "snippet_input" else empty_01_np

        arrs = [timing_t.numpy(), cols_t.numpy()] + [t.numpy() for t in optional_t]
        # Dict vs non-dict must follow use_dict_output: step_index can be empty while intervals/other optionals exist.
        if not use_dict_output:
            feats_out, cols_out = _apply_timing_jitter_py_callback(
                arrs[0],
                arrs[1],
                sigma,
                False,
                use_interval,
                interval_encoding,
            )
            return (feats_out, cols_out) + tuple(
                _empty(k) for k in _JITTER_OPTIONAL_KEYS
            )
        # Dict path: build feature dict, jitter, then return flat list again.
        feats_dict: dict[str, np.ndarray] = {"timing_input": arrs[0]}
        for i, key in enumerate(_JITTER_OPTIONAL_KEYS):
            if arrs[i + 2].size > 0:
                feats_dict[key] = arrs[i + 2]
        feats_out, cols_out = _apply_timing_jitter_py_callback(
            feats_dict,
            arrs[1],
            sigma,
            True,
            use_interval,
            interval_encoding,
        )
        return (feats_out["timing_input"], cols_out) + tuple(
            feats_out.get(k, _empty(k)) for k in _JITTER_OPTIONAL_KEYS
        )

    res = tf.py_function(  # type: ignore[misc]
        _py_jitter,
        flat,
        (tf.float32, tf.int32) + (tf.float32,) * len(_JITTER_OPTIONAL_KEYS),
    )
    out_timing = tf.ensure_shape(res[0], flat[0].shape)
    out_cols = tf.ensure_shape(res[1], flat[1].shape)
    if use_dict_output:
        out_dict: dict[str, tf.Tensor] = {"timing_input": out_timing}
        for i, key in enumerate(_JITTER_OPTIONAL_KEYS):
            t = res[i + 2]
            # Include keys that were present in the original features dict OR
            # were created by the jitter callback (non-empty tensor).
            if key in features or t.shape[0] != 0:
                t = (
                    tf.ensure_shape(t, (None, n_frames_window, _N_MELS))
                    if key == "snippet_input"
                    else tf.ensure_shape(t, (None, 1))
                )
                out_dict[key] = t
        return out_dict, out_cols
    return out_timing, out_cols


def _process_arrow_pair_tf_map(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    snippet_half_frames: int,
    use_interval: bool,
    interval_encoding: config.IntervalEncoding,
    use_step_index: bool,
    use_beat_phase: bool,
    use_aux_interval_target: bool,
) -> tuple[tf.Tensor | dict[str, tf.Tensor], tf.Tensor]:
    """Map (audio_path, chart_path) to arrow inputs and cols.

    Uses _load_arrow_pair_py_callback; builds dict or (times, cols) from the
    returned 8-tuple based on config flags.
    """
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
    ) = tf.py_function(  # type: ignore[misc]
        lambda ap, cp: _load_arrow_pair_py_callback(
            ap,
            cp,
            snippet_half_frames,
            use_interval,
            interval_encoding,
            use_step_index,
            use_beat_phase,
            use_aux_interval_target,
        ),
        [audio_path_t, chart_path_t],
        (
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.int32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
        ),
    )
    times = tf.ensure_shape(times, [None])
    times = tf.expand_dims(times, axis=-1)
    cols = tf.ensure_shape(cols, [None])

    use_snippets = snippet_half_frames > 0
    use_dict = _arrow_use_dict_output(
        snippet_half_frames,
        use_interval,
        use_step_index,
        use_beat_phase,
        use_aux_interval_target,
    )

    if use_dict:
        out: dict[str, tf.Tensor] = {"timing_input": times}
        if use_interval:
            intervals = tf.ensure_shape(intervals, [None])
            if interval_encoding == config.IntervalEncoding.DEFAULT:
                out["interval_input"] = tf.expand_dims(intervals, axis=-1)
            elif interval_encoding == config.IntervalEncoding.LOG:
                out["interval_log_input"] = tf.expand_dims(intervals, axis=-1)
            elif interval_encoding == config.IntervalEncoding.MULTI:
                interval_log = tf.ensure_shape(interval_log, [None])
                out["interval_log_input"] = tf.expand_dims(interval_log, axis=-1)
                out["interval_next_input"] = tf.expand_dims(intervals, axis=-1)
        if use_step_index:
            step_index = tf.ensure_shape(step_index, [None])
            out["step_index_input"] = tf.expand_dims(step_index, axis=-1)
        if use_beat_phase:
            beat_phase = tf.ensure_shape(beat_phase, [None])
            out["beat_phase_input"] = tf.expand_dims(beat_phase, axis=-1)
        if use_snippets:
            n_frames_window = 2 * snippet_half_frames + 1
            snippets = tf.ensure_shape(snippets, [None, n_frames_window, _N_MELS])
            out["snippet_input"] = snippets
        if use_aux_interval_target:
            aux_interval = tf.ensure_shape(aux_interval, [None])
            out["aux_interval_target"] = tf.expand_dims(aux_interval, axis=-1)
            aux_interval_mask = tf.ensure_shape(aux_interval_mask, [None])
            out["aux_interval_mask"] = tf.expand_dims(aux_interval_mask, axis=-1)
        return out, cols

    return times, cols


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
) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset pipeline with a proper caching strategy.
    Deterministic preprocessing is cached, while random augmentations are applied
    on the fly in each epoch.
    """
    if n_features is None:
        if feature_source == config.FeatureSource.MERT:
            n_features = constants.MERT_HIDDEN_SIZE
        else:
            n_features = _N_MELS

    pairs = _load_and_pair_files(data_dir)
    if not pairs:
        raise ValueError("No audio-chart pairs found.")

    ds = tf.data.Dataset.from_tensor_slices(pairs)
    ds = ds.map(
        lambda p: _load_and_preprocess_tf_map(
            p[0],
            p[1],
            use_gaussian_target,
            gaussian_sigma,
            feature_source,
            mert_features_dir,
            data_dir,
            n_features,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.cache()
    ds = ds.map(
        lambda features, target: _apply_augmentations_tf_map(
            features,
            target,
            apply_temporal_augment,
            should_apply_spec_augment,
            n_features,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if batch_size > 1:
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(
                ((None, n_features)),
                (None, 1),
            ),
            padding_values=(
                (0.0),
                0.0,
            ),
        )
    else:
        ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset for arrow prediction.

    Step times are always normalized (critical for training and inference).
    Sequences are padded to constants.MAX_STEPS so batches have fixed shape (required for XLA).
    When timing_jitter_sigma > 0, Gaussian jitter is applied to timing_input after cache
    (uncached map) so each epoch sees new noise; validation should use timing_jitter_sigma=0.

    Args:
        data_dir: Directory containing audio and chart files.
        batch_size: Number of samples per batch.
        snippet_half_frames: Half-window of frames around each onset (total = 2*snippet_half_frames+1).
            When > 0, load audio and yield mel snippets per step; when 0, timing only.
        use_interval: If True, include interval_input (time since previous step) in the batch dict.
        interval_encoding: IntervalEncoding (DEFAULT, LOG, or MULTI). Must match model config.
        use_step_index: If True, include step_index_input (normalized position in sequence).
        use_beat_phase: If True, include beat_phase_input (BPM from chart txt).
        use_aux_interval_target: If True, include aux_interval_target (next-step interval) for aux loss.
        timing_jitter_sigma: If > 0, add Gaussian jitter to timing_input (training only).
            0 disables jitter. Apply 0 for validation.

    Returns:
        Dataset yielding (dict of inputs/targets, cols) when any extra feature is used,
        else (times, cols). Dict may contain timing_input, interval_input, interval_log_input,
        interval_next_input, step_index_input, beat_phase_input, snippet_input, aux_interval_target.
    """
    pairs = _load_and_pair_files(data_dir)
    if not pairs:
        raise ValueError("No audio-chart pairs found in the specified directory.")

    ds = tf.data.Dataset.from_tensor_slices(pairs)
    n_frames_window = 2 * snippet_half_frames + 1

    ds = ds.map(
        lambda pair: _process_arrow_pair_tf_map(
            pair[0],
            pair[1],
            snippet_half_frames,
            use_interval,
            interval_encoding,
            use_step_index,
            use_beat_phase,
            use_aux_interval_target,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.cache()

    use_dict_output = _arrow_use_dict_output(
        snippet_half_frames,
        use_interval,
        use_step_index,
        use_beat_phase,
        use_aux_interval_target,
    )

    if timing_jitter_sigma > 0:
        ds = ds.map(
            lambda feats, c: _apply_timing_jitter_tf_map(
                feats,
                c,
                timing_jitter_sigma,
                use_dict_output,
                use_interval,
                interval_encoding,
                n_frames_window,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if batch_size > 1:
        if use_dict_output:
            padded_shapes_dict: dict[str, tuple[Any, ...]] = {"timing_input": (None, 1)}
            padding_values_dict: dict[str, float | int] = {"timing_input": 0.0}
            if snippet_half_frames > 0:
                padded_shapes_dict["snippet_input"] = (
                    None,
                    n_frames_window,
                    _N_MELS,
                )
                padding_values_dict["snippet_input"] = 0.0
            if use_interval:
                if interval_encoding == config.IntervalEncoding.DEFAULT:
                    padded_shapes_dict["interval_input"] = (None, 1)
                    padding_values_dict["interval_input"] = 0.0
                elif interval_encoding == config.IntervalEncoding.LOG:
                    padded_shapes_dict["interval_log_input"] = (None, 1)
                    padding_values_dict["interval_log_input"] = 0.0
                elif interval_encoding == config.IntervalEncoding.MULTI:
                    padded_shapes_dict["interval_log_input"] = (None, 1)
                    padding_values_dict["interval_log_input"] = 0.0
                    padded_shapes_dict["interval_next_input"] = (None, 1)
                    padding_values_dict["interval_next_input"] = 0.0
            if use_step_index:
                padded_shapes_dict["step_index_input"] = (None, 1)
                padding_values_dict["step_index_input"] = 0.0
            if use_beat_phase:
                padded_shapes_dict["beat_phase_input"] = (None, 1)
                padding_values_dict["beat_phase_input"] = 0.0
            if use_aux_interval_target:
                padded_shapes_dict["aux_interval_target"] = (None, 1)
                padding_values_dict["aux_interval_target"] = 0.0
                padded_shapes_dict["aux_interval_mask"] = (None, 1)
                padding_values_dict["aux_interval_mask"] = 0.0
            ds = ds.padded_batch(
                batch_size,
                padded_shapes=(padded_shapes_dict, (None,)),
                padding_values=(padding_values_dict, constants.ARROW_PADDING_CLASS),
            )
        else:
            ds = ds.padded_batch(
                batch_size,
                padded_shapes=((None, 1), (None,)),
                padding_values=(0.0, constants.ARROW_PADDING_CLASS),
            )
    else:
        ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
