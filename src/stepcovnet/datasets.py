"""Data collection and preprocessing for StepCovNet.

This module provides functionality to load audio and StepMania chart files,
process them into spectrograms and target vectors, and create a TensorFlow
dataset for training.
"""

import os
import pathlib
from typing import Any

import librosa
import numpy as np
import tensorflow as tf
from scipy import interpolate

from stepcovnet import constants

HOP_COEFF = 0.01  # 100ms per frame

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
    pairs = []
    for root, _, files in os.walk(data_dir):
        audio_files = [f for f in files if f.endswith((".mp3", ".ogg", ".wav"))]
        chart_files = [f for f in files if f.endswith(".txt")]

        # Pair files with same stem (e.g., 'song.mp3' and 'song.sm')
        for audio_file in audio_files:
            stem = pathlib.Path(audio_file).stem
            matching_charts = [f for f in chart_files if f.startswith(stem)]
            if matching_charts:
                pairs.append(
                    (
                        os.path.join(root, audio_file),
                        os.path.join(root, matching_charts[0]),
                    )
                )
    return pairs


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
    with open(chart_path) as f:
        f.readline()  # TITLE
        _ = float(f.readline().removeprefix("BPM").strip())  # BPM
        f.readline()  # NOTES
        difficulty_level = f.readline().strip().lower().split(" ")[1]
        _ = _DIFFICULTY_MAP.get(difficulty_level, 2)
        times = []
        cols = []
        for line in f:
            if line.startswith("DIFFICULTY"):
                # TODO: Read off of multiple difficulties
                break
            # TODO: Use the type of note played and not just the presence
            arrows, timing = line.strip().split(" ")
            times.append(float(timing))
            if binary_timings:
                cols.append(0)
            else:
                cols.append(_base4_to_int(arrows))

    return np.array(times), np.array(cols, dtype=np.int32)


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


def _temporal_augment_scipy(
    spec: np.ndarray, labels_and_features: np.ndarray
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

    spec_resized = np.zeros((_N_MELS, new_length), dtype=spec.dtype)
    original_time = np.arange(original_length)
    warped_time = np.linspace(0, original_length - 1, new_length)

    for bin_idx in range(_N_MELS):
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
) -> tuple[np.ndarray, np.ndarray]:
    """Load audio and chart, build features and target (pure Python, no TF)."""
    spec = audio_to_spectrogram(audio_path)
    spec_length = spec.shape[1]
    times, cols = _parse_step_chart(chart_path, binary_timings=True)
    target = (
        _create_target_gaussian(times, cols, spec_length, gaussian_sigma)
        if use_gaussian_target
        else _create_target(times, cols, spec_length)
    )
    features = np.transpose(spec)
    return features.astype(np.float32), target.astype(np.float32)


def _load_and_preprocess_py_callback(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    use_gaussian_target: bool,
    gaussian_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode paths and delegate to _load_and_preprocess_paths (for tf.py_function)."""
    audio_path = audio_path_t.numpy().decode()  # type: ignore[union-attr]
    chart_path = chart_path_t.numpy().decode()  # type: ignore[union-attr]
    return _load_and_preprocess_paths(
        audio_path, chart_path, use_gaussian_target, gaussian_sigma
    )


def _load_and_preprocess_tf_map(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    use_gaussian_target: bool,
    gaussian_sigma: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map one (audio_path, chart_path) to (features, target) tensors."""
    features, target = tf.py_function(  # type: ignore[misc]
        lambda ap, cp: _load_and_preprocess_py_callback(
            ap, cp, use_gaussian_target, gaussian_sigma
        ),
        [audio_path_t, chart_path_t],
        (tf.float32, tf.float32),
    )
    features.set_shape([None, _N_MELS])
    target.set_shape([None, _N_TARGET])
    return features, target


def _augment_features_numpy(
    features: np.ndarray,
    target: np.ndarray,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply optional temporal/spec augmentation and normalize (pure Python)."""
    spec_py = np.transpose(features[:, :_N_MELS])
    combined_labels = target
    if apply_temporal_augment:
        spec_py, combined_labels = _temporal_augment_scipy(spec_py, combined_labels)
    spec_py = normalize_onset_spectrogram(spec_py.T).T
    if should_apply_spec_augment:
        spec_py = _apply_spec_augment(spec_py, F=int(0.2 * _N_MELS))
    final_target = combined_labels[:, :_N_TARGET]
    final_features = np.transpose(spec_py)
    return final_features.astype(np.float32), final_target.astype(np.float32)


def _augment_py_callback(
    features_t: tf.Tensor,
    target_t: tf.Tensor,
    temp_aug: bool,
    spec_aug: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert tensors to numpy and delegate to _augment_features_numpy."""
    features = features_t.numpy()  # type: ignore[union-attr]
    target = target_t.numpy()  # type: ignore[union-attr]
    return _augment_features_numpy(features, target, temp_aug, spec_aug)


def _apply_augmentations_tf_map(
    features: tf.Tensor,
    target: tf.Tensor,
    apply_temporal_augment: bool,
    should_apply_spec_augment: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map (features, target) to augmented (features, target) tensors."""
    aug_features, aug_target = tf.py_function(  # type: ignore[misc]
        _augment_py_callback,
        [features, target, apply_temporal_augment, should_apply_spec_augment],
        (tf.float32, tf.float32),
    )
    aug_features.set_shape([None, _N_MELS])
    aug_target.set_shape([None, _N_TARGET])
    return aug_features, aug_target


def _load_arrow_pair_py_callback(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    snippet_half_frames: int,
    use_interval: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode (audio_path, chart_path) tensors and load arrow data for tf.py_function.

    Parses chart, optionally loads audio and extracts mel snippets when
    snippet_half_frames > 0, and optionally computes normalized intervals when
    use_interval is True. Always returns four arrays so tf.py_function has a
    fixed signature; when snippet_half_frames is 0, audio is not loaded and
    snippets are (n_steps, 0, n_mels).

    Returns:
        times: (n_steps,) float32, normalized to [0, 1].
        intervals: (n_steps,) float32, normalized when use_interval else zeros.
        snippets: (n_steps, n_frames, n_mels) when snippet_half_frames > 0, else
            (n_steps, 0, n_mels).
        cols: (n_steps,) int32.
    """
    audio_path = audio_path_t.numpy().decode()  # type: ignore[union-attr]
    chart_path = chart_path_t.numpy().decode()  # type: ignore[union-attr]

    times, cols = _parse_step_chart(chart_path, binary_timings=False)
    times = np.asarray(times, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.int32)
    n_steps = len(times)
    n_frames = (2 * snippet_half_frames + 1) if snippet_half_frames > 0 else 0

    if n_steps == 0:
        return (
            times.astype(np.float32),
            times.astype(np.float32),
            np.zeros((0, n_frames, _N_MELS), dtype=np.float32),
            cols,
        )

    times_norm = (times / (np.max(times) + 1e-9)).astype(np.float32)
    intervals_norm = (
        normalized_intervals_from_times(times)
        if use_interval
        else np.zeros(n_steps, dtype=np.float32)
    )

    if snippet_half_frames > 0:
        spec_time_major = normalize_onset_spectrogram(
            audio_to_spectrogram(audio_path).T
        )
        snippets = extract_snippets_from_spec(
            spec_time_major, times, snippet_half_frames
        )
    else:
        snippets = np.zeros((n_steps, 0, _N_MELS), dtype=np.float32)

    return times_norm, intervals_norm, snippets, cols


def _process_arrow_pair_tf_map(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    snippet_half_frames: int,
    use_interval: bool,
) -> tuple[tf.Tensor | dict[str, tf.Tensor], tf.Tensor]:
    """Map (audio_path, chart_path) to arrow inputs and cols.

    Uses _load_arrow_pair_py_callback for all cases; builds dict or (times, cols)
    from the fixed (times, intervals, snippets, cols) return based on
    snippet_half_frames and use_interval.
    """
    times, intervals, snippets, cols = tf.py_function(  # type: ignore[misc]
        lambda ap, cp: _load_arrow_pair_py_callback(
            ap, cp, snippet_half_frames, use_interval
        ),
        [audio_path_t, chart_path_t],
        (tf.float32, tf.float32, tf.float32, tf.int32),
    )
    times = tf.ensure_shape(times, [None])
    times = tf.expand_dims(times, axis=-1)
    cols = tf.ensure_shape(cols, [None])

    use_snippets = snippet_half_frames > 0
    if use_snippets or use_interval:
        out = {"timing_input": times}
        if use_interval:
            intervals = tf.ensure_shape(intervals, [None])
            intervals = tf.expand_dims(intervals, axis=-1)
            out["interval_input"] = intervals
        if use_snippets:
            n_frames_window = 2 * snippet_half_frames + 1
            snippets = tf.ensure_shape(snippets, [None, n_frames_window, _N_MELS])
            out["snippet_input"] = snippets
        return out, cols

    return times, cols


def create_dataset(
    data_dir: str,
    batch_size: int = 1,
    apply_temporal_augment: bool = False,
    should_apply_spec_augment: bool = False,
    use_gaussian_target: bool = False,
    gaussian_sigma: float = 1.0,
) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset pipeline with a proper caching strategy.
    Deterministic preprocessing is cached, while random augmentations are applied
    on the fly in each epoch.
    """
    pairs = _load_and_pair_files(data_dir)
    if not pairs:
        raise ValueError("No audio-chart pairs found.")

    ds = tf.data.Dataset.from_tensor_slices(pairs)
    ds = ds.map(
        lambda p: _load_and_preprocess_tf_map(
            p[0], p[1], use_gaussian_target, gaussian_sigma
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
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if batch_size > 1:
        ds = ds.padded_batch(
            batch_size,
            padded_shapes=(
                ((None, _N_MELS)),  # Spectrogram (mel bands x time)
                (None, 1),  # Target (time x columns)
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
) -> tf.data.Dataset:
    """Creates a TensorFlow dataset for arrow prediction.

    Step times are always normalized (critical for training and inference).
    Sequences are padded to constants.MAX_STEPS so batches have fixed shape (required for XLA).

    Args:
        data_dir: Directory containing audio and chart files.
        batch_size: Number of samples per batch.
        snippet_half_frames: Half-window of frames around each onset (total = 2*snippet_half_frames+1).
            When > 0, load audio and yield mel snippets per step; when 0, timing only.
        use_interval: If True, include interval_input (time since previous step) in the batch dict.

    Returns:
        When snippet_half_frames=0 and use_interval=False: dataset yielding (times, cols).
        When snippet_half_frames=0 and use_interval=True: dataset yielding (dict with timing_input,
            interval_input, cols).
        When snippet_half_frames>0: dataset yielding (dict with timing_input, snippet_input,
            optionally interval_input, cols).
    """
    pairs = _load_and_pair_files(data_dir)
    if not pairs:
        raise ValueError("No audio-chart pairs found in the specified directory.")

    ds = tf.data.Dataset.from_tensor_slices(pairs)
    n_frames_window = 2 * snippet_half_frames + 1

    ds = ds.map(
        lambda pair: _process_arrow_pair_tf_map(
            pair[0], pair[1], snippet_half_frames, use_interval
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.cache()

    if batch_size > 1:
        use_dict_output = snippet_half_frames > 0 or use_interval
        if use_dict_output:
            padded_shapes_dict: dict[str, tuple[Any, ...]] = {"timing_input": (None, 1)}
            padding_values_dict = {"timing_input": 0.0}
            if snippet_half_frames > 0:
                padded_shapes_dict["snippet_input"] = (None, n_frames_window, _N_MELS)
                padding_values_dict["snippet_input"] = 0.0
            if use_interval:
                padded_shapes_dict["interval_input"] = (None, 1)
                padding_values_dict["interval_input"] = 0.0
            ds = ds.padded_batch(
                batch_size,
                padded_shapes=(padded_shapes_dict, (None,)),
                padding_values=(padding_values_dict, 0),
            )
        else:
            ds = ds.padded_batch(
                batch_size,
                padded_shapes=((None, 1), (None,)),
                padding_values=(0.0, 0),
            )
    else:
        ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
