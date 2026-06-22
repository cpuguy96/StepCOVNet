"""TensorFlow dataset pipeline for event-based onset detection."""

import dataclasses
import pathlib

import numpy as np
import tensorflow as tf

from stepcovnet import constants, pairing
from stepcovnet.onset_events import audio, charts, preprocess, targets


@dataclasses.dataclass
class OnsetEventDatasetConfig:
    """Configuration for ``create_onset_event_dataset``.

    Attributes:
        batch_size: Number of samples per batch.
        max_audio_seconds: Maximum audio duration in seconds before truncation.
        n_max_onsets: Fixed length for padded ground-truth onset times.
        max_steps_per_chart: Skip charts with more than this many steps.
        target_sample_rate: Sample rate in Hz for waveform loading.
        frontend: Pre-processing type (``conv1d``, ``mel``, or ``mert``).
        mert_features_dir: Directory of precomputed MERT features.
        data_root: Training data root for nested MERT paths.
    """

    batch_size: int = 1
    max_audio_seconds: float = audio.DEFAULT_MAX_AUDIO_SECONDS
    n_max_onsets: int = targets.N_MAX_ONSETS
    max_steps_per_chart: int = charts.MAX_STEPS_PER_CHART
    target_sample_rate: int = constants.TARGET_SR
    frontend: str = preprocess.FRONTEND_CONV1D
    mert_features_dir: str = ""
    data_root: str = ""


def _max_samples(config: OnsetEventDatasetConfig) -> int:
    return audio.max_samples_for_cap(
        config.max_audio_seconds,
        config.target_sample_rate,
    )


def first_valid_pair(
    data_dir: str,
    *,
    max_steps_per_chart: int = charts.MAX_STEPS_PER_CHART,
) -> tuple[str, str]:
    """Return the first valid audio/chart pair under ``data_dir``.

    Args:
        data_dir: Root directory searched for paired audio and chart files.
        max_steps_per_chart: Skip charts with more than this many steps.

    Returns:
        ``(audio_path, chart_path)`` for the first pair that passes filtering.

    Raises:
        ValueError: When no valid pair is found.
    """
    pairs = _filter_valid_pairs(
        pairing.list_audio_chart_pairs(data_dir),
        max_steps_per_chart,
    )
    if not pairs:
        raise ValueError(f"No valid audio-chart pairs found under {data_dir!r}")
    return pairs[0]


def _filter_valid_pairs(
    pairs: list[tuple[str, str]],
    max_steps_per_chart: int,
) -> list[tuple[str, str]]:
    """Keep pairs with existing files and charts within the step cap."""
    valid: list[tuple[str, str]] = []
    for audio_path, chart_path in pairs:
        if (
            not pathlib.Path(audio_path).is_file()
            or not pathlib.Path(chart_path).is_file()
        ):
            continue
        if charts.chart_exceeds_step_cap(chart_path, max_steps=max_steps_per_chart):
            continue
        if charts.load_onset_times(chart_path, max_steps=max_steps_per_chart) is None:
            continue
        valid.append((audio_path, chart_path))
    return valid


def _load_onset_event_sample(
    audio_path: str,
    chart_path: str,
    *,
    target_sample_rate: int,
    max_samples: int,
    max_audio_seconds: float,
    n_max_onsets: int,
    max_steps_per_chart: int,
    frontend: str = preprocess.FRONTEND_CONV1D,
    mert_features_dir: str = "",
    data_root: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one audio/chart pair into numpy arrays for the training batch."""
    gt_times = charts.load_onset_times(chart_path, max_steps=max_steps_per_chart)
    if gt_times is None:
        raise ValueError(f"chart exceeds step cap: {chart_path}")

    encoder_input, audio_length, duration = preprocess.load_preprocessed_encoder_input(
        audio_path,
        frontend_name=frontend,
        target_sample_rate=target_sample_rate,
        max_samples=max_samples,
        max_audio_seconds=max_audio_seconds,
        mert_features_dir=mert_features_dir,
        data_root=data_root,
    )

    gt_times = targets.clip_times_to_duration(gt_times, float(duration))
    gt_times_padded, gt_mask = targets.pad_onset_times(gt_times, n_max=n_max_onsets)

    max_frames = preprocess.max_encoder_frames(max_audio_seconds)
    feature_dim = preprocess.encoder_feature_dim(frontend)
    if frontend == preprocess.FRONTEND_CONV1D:
        audio_padded = encoder_input.astype(np.float32, copy=False)
        features = np.zeros((max_frames, feature_dim), dtype=np.float32)
    else:
        audio_padded = np.zeros(max_samples, dtype=np.float32)
        features = encoder_input.astype(np.float32, copy=False)

    return (
        audio_padded,
        np.asarray(audio_length, dtype=np.int32),
        gt_times_padded,
        gt_mask,
        np.asarray(duration, dtype=np.float32),
        features,
    )


def _load_onset_event_py_callback(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    target_sample_rate: int,
    max_samples: int,
    max_audio_seconds: float,
    n_max_onsets: int,
    max_steps_per_chart: int,
    frontend: str,
    mert_features_dir: str,
    data_root: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Decode path tensors and load one sample (for ``tf.py_function``)."""
    audio_path = audio_path_t.numpy().decode()  # type: ignore[union-attr]
    chart_path = chart_path_t.numpy().decode()  # type: ignore[union-attr]
    (
        audio_t,
        audio_length_t,
        gt_times_t,
        gt_mask_t,
        duration_t,
        features_t,
    ) = _load_onset_event_sample(
        audio_path,
        chart_path,
        target_sample_rate=target_sample_rate,
        max_samples=max_samples,
        max_audio_seconds=max_audio_seconds,
        n_max_onsets=n_max_onsets,
        max_steps_per_chart=max_steps_per_chart,
        frontend=frontend,
        mert_features_dir=mert_features_dir,
        data_root=data_root,
    )
    return audio_t, audio_length_t, gt_times_t, gt_mask_t, duration_t, features_t


def _map_pair_to_sample(
    audio_path_t: tf.Tensor,
    chart_path_t: tf.Tensor,
    config: OnsetEventDatasetConfig,
    max_samples: int,
) -> dict[str, tf.Tensor]:
    """Map one (audio_path, chart_path) pair to a training sample dict."""
    max_frames = preprocess.max_encoder_frames(config.max_audio_seconds)
    feature_dim = preprocess.encoder_feature_dim(config.frontend)
    (
        audio_t,
        audio_length_t,
        gt_times_t,
        gt_mask_t,
        duration_t,
        features_t,
    ) = tf.py_function(  # type: ignore[misc]
        lambda ap, cp: _load_onset_event_py_callback(
            ap,
            cp,
            config.target_sample_rate,
            max_samples,
            config.max_audio_seconds,
            config.n_max_onsets,
            config.max_steps_per_chart,
            config.frontend,
            config.mert_features_dir,
            config.data_root,
        ),
        [audio_path_t, chart_path_t],
        (
            tf.float32,
            tf.int32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
        ),
    )
    audio_t.set_shape([max_samples])
    audio_length_t.set_shape([])
    gt_times_t.set_shape([config.n_max_onsets])
    gt_mask_t.set_shape([config.n_max_onsets])
    duration_t.set_shape([])
    features_t.set_shape([max_frames, feature_dim])
    sample = {
        "audio": audio_t,
        "audio_length": audio_length_t,
        "gt_times": gt_times_t,
        "gt_mask": gt_mask_t,
        "duration": duration_t,
        "features": features_t,
    }
    return sample


def create_onset_event_dataset_from_pairs(
    pairs: list[tuple[str, str]],
    *,
    batch_size: int = 1,
    max_audio_seconds: float = audio.DEFAULT_MAX_AUDIO_SECONDS,
    n_max_onsets: int = targets.N_MAX_ONSETS,
    max_steps_per_chart: int = charts.MAX_STEPS_PER_CHART,
    target_sample_rate: int = constants.TARGET_SR,
    shuffle: bool = False,
    seed: int | None = None,
    frontend: str = preprocess.FRONTEND_CONV1D,
    mert_features_dir: str = "",
    data_root: str = "",
) -> tf.data.Dataset:
    """Build a ``tf.data`` pipeline from explicit audio/chart path pairs.

    Args:
        pairs: List of ``(audio_path, chart_path)`` tuples.
        batch_size: Number of samples per batch.
        max_audio_seconds: Maximum audio duration in seconds before truncation.
        n_max_onsets: Fixed length for padded ground-truth onset times.
        max_steps_per_chart: Skip charts with more than this many steps.
        target_sample_rate: Sample rate in Hz for waveform loading.
        shuffle: Whether to shuffle pair order each epoch (ignored when only one pair).
        seed: Random seed used when ``shuffle`` is True.

    Returns:
        Dataset yielding dict batches with keys ``audio``, ``audio_length``,
        ``gt_times``, ``gt_mask``, and ``duration``.

    Raises:
        ValueError: When ``pairs`` is empty or no pair remains after filtering.
    """
    pipeline_config = OnsetEventDatasetConfig(
        batch_size=batch_size,
        max_audio_seconds=max_audio_seconds,
        n_max_onsets=n_max_onsets,
        max_steps_per_chart=max_steps_per_chart,
        target_sample_rate=target_sample_rate,
        frontend=frontend,
        mert_features_dir=mert_features_dir,
        data_root=data_root,
    )
    max_samples = _max_samples(pipeline_config)

    valid_pairs = _filter_valid_pairs(pairs, pipeline_config.max_steps_per_chart)
    if not valid_pairs:
        raise ValueError("No valid audio-chart pairs found.")

    ds = tf.data.Dataset.from_tensor_slices(valid_pairs)
    if shuffle and len(valid_pairs) > 1:
        ds = ds.shuffle(buffer_size=len(valid_pairs), seed=seed)

    ds = ds.map(
        lambda pair: _map_pair_to_sample(
            pair[0], pair[1], pipeline_config, max_samples
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.cache()
    ds = ds.batch(pipeline_config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def create_onset_event_dataset(
    data_dir: str,
    *,
    batch_size: int = 1,
    max_audio_seconds: float = audio.DEFAULT_MAX_AUDIO_SECONDS,
    n_max_onsets: int = targets.N_MAX_ONSETS,
    max_steps_per_chart: int = charts.MAX_STEPS_PER_CHART,
    target_sample_rate: int = constants.TARGET_SR,
    shuffle: bool = False,
    seed: int | None = None,
    frontend: str = preprocess.FRONTEND_CONV1D,
    mert_features_dir: str = "",
    data_root: str = "",
) -> tf.data.Dataset:
    """Build a ``tf.data`` pipeline over audio/chart pairs for event onset training.

    Pairs are skipped when files are missing, the chart exceeds ``max_steps_per_chart``,
    or ``load_onset_times`` returns ``None``. Waveforms are truncated to
    ``max_audio_seconds``, ground-truth times are clipped to the resulting duration,
    then audio and times are padded to fixed shapes.

    Args:
        data_dir: Root directory searched for paired audio and chart files.
        batch_size: Number of samples per batch.
        max_audio_seconds: Maximum audio duration in seconds before truncation.
        n_max_onsets: Fixed length for padded ground-truth onset times.
        max_steps_per_chart: Skip charts with more than this many steps.
        target_sample_rate: Sample rate in Hz for waveform loading.
        shuffle: Whether to shuffle pair order each epoch.
        seed: Random seed used when ``shuffle`` is True.

    Returns:
        Dataset yielding dict batches with keys ``audio``, ``audio_length``,
        ``gt_times``, ``gt_mask``, and ``duration``.

    Raises:
        ValueError: When no valid audio/chart pairs remain after filtering.
    """
    pairs = _filter_valid_pairs(
        pairing.list_audio_chart_pairs(data_dir),
        max_steps_per_chart,
    )
    if not pairs:
        raise ValueError("No valid audio-chart pairs found.")

    return create_onset_event_dataset_from_pairs(
        pairs,
        batch_size=batch_size,
        max_audio_seconds=max_audio_seconds,
        n_max_onsets=n_max_onsets,
        max_steps_per_chart=max_steps_per_chart,
        target_sample_rate=target_sample_rate,
        shuffle=shuffle,
        seed=seed,
    )
