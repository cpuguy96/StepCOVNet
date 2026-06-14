"""Inference API: raw audio to filtered onset times and confidences."""

import os

import numpy as np

from stepcovnet.onset_events import audio
from stepcovnet.onset_events import metrics


def _model_has_duration_input(model) -> bool:
    """Return True when the Keras model expects a ``duration`` input."""
    inputs = model.inputs
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    for model_input in inputs:
        name = model_input.name.split(":")[0]
        if name == "duration":
            return True
    return False


def _resolve_waveform(
    audio_path_or_waveform: str | os.PathLike[str] | np.ndarray,
    *,
    target_sample_rate: int,
) -> np.ndarray:
    """Load or validate a mono waveform for inference.

    Args:
        audio_path_or_waveform: Path to an audio file or a one-dimensional
            waveform array.
        target_sample_rate: Sample rate in Hz for file loading.

    Returns:
        One-dimensional float32 waveform.

    Raises:
        ValueError: If a waveform array is not one-dimensional.
        TypeError: If the input type is unsupported.
    """
    if isinstance(audio_path_or_waveform, (str, os.PathLike)):
        return audio.load_waveform(
            os.fspath(audio_path_or_waveform),
            target_sample_rate=target_sample_rate,
        )

    waveform = np.asarray(audio_path_or_waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(
            f"waveform must be one-dimensional; got shape {waveform.shape}"
        )
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return np.asarray(waveform, dtype=np.float32)


def _prepare_waveform_batch(
    waveform: np.ndarray,
    *,
    max_samples: int,
    target_sample_rate: int,
) -> tuple[np.ndarray, float]:
    """Truncate, pad, and compute duration like the training dataset loader.

    Args:
        waveform: One-dimensional float32 waveform.
        max_samples: Maximum number of samples after truncation and padding.
        target_sample_rate: Sample rate in Hz used for duration conversion.

    Returns:
        Tuple of ``(padded_audio_batch, duration_sec)`` where the batch has
        shape ``(1, max_samples)``.
    """
    truncated = audio.truncate_waveform(waveform, max_samples)
    duration_sec = float(truncated.size) / float(target_sample_rate)
    padded = audio.pad_waveform(truncated, max_samples)
    return padded[np.newaxis, :], duration_sec


def _extract_prediction_arrays(outputs, model) -> tuple[np.ndarray, np.ndarray]:
    """Normalize ``model.predict`` output to one batch of times and confidence."""
    if isinstance(outputs, dict):
        pred_times = outputs["pred_times"]
        pred_confidence = outputs["pred_confidence"]
    else:
        output_names = list(model.output_names)
        outputs_by_name = {
            name: array for name, array in zip(output_names, outputs, strict=True)
        }
        pred_times = outputs_by_name["pred_times"]
        pred_confidence = outputs_by_name["pred_confidence"]

    times = np.asarray(pred_times[0], dtype=np.float32)
    confidence = np.asarray(pred_confidence[0], dtype=np.float32)
    return times, confidence


def _apply_confidence_threshold(
    times_sec: np.ndarray,
    confidences: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep predictions at or above ``confidence_threshold``."""
    return metrics.filter_predicted_onsets_numpy(
        times_sec,
        confidences,
        confidence_threshold,
        min_onset_distance_ms=0.0,
    )


def _apply_min_onset_distance(
    times_sec: np.ndarray,
    confidences: np.ndarray,
    min_onset_distance_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by time and drop pairs closer than ``min_onset_distance_ms``.

    When two predictions fall within the minimum gap, the earlier time is kept
    (mirrors time-ordered min-gap filtering used for dense onset post-processing).
    """
    return metrics.filter_predicted_onsets_numpy(
        times_sec,
        confidences,
        confidence_threshold=0.0,
        min_onset_distance_ms=min_onset_distance_ms,
    )


def predict_onsets(
    model,
    audio_path_or_waveform: str | os.PathLike[str] | np.ndarray,
    *,
    confidence_threshold: float = 0.5,
    min_onset_distance_ms: float = 50.0,
    target_sample_rate: int = 44100,
    max_audio_seconds: float = audio.DEFAULT_MAX_AUDIO_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run event onset inference on raw audio.

    Loads audio from a file path or accepts a one-dimensional waveform, applies
    the same truncate-and-pad preprocessing as training, runs ``model.predict``,
    filters by confidence, sorts by time, and enforces a minimum gap between
    onsets.

    Args:
        model: Trained Keras onset event model with outputs ``pred_times`` and
            ``pred_confidence``. When the model includes a ``duration`` input,
            ``pred_times`` are returned in seconds; otherwise they are scaled
            by the truncated audio duration.
        audio_path_or_waveform: Path to an audio file or a one-dimensional
            waveform array.
        confidence_threshold: Minimum confidence in ``[0, 1]`` to keep a slot.
        min_onset_distance_ms: Minimum time separation between kept onsets in
            milliseconds.
        target_sample_rate: Sample rate in Hz for loading and duration math.
        max_audio_seconds: Maximum audio duration before truncation.

    Returns:
        Tuple of ``(times_sec, confidences)`` as one-dimensional float32 arrays
        sorted by time with the same length.

    Raises:
        ValueError: If threshold or gap parameters are invalid, or the waveform
            shape is wrong.
        TypeError: If ``audio_path_or_waveform`` has an unsupported type.
    """
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if min_onset_distance_ms < 0.0:
        raise ValueError("min_onset_distance_ms must be non-negative")

    max_samples = audio.max_samples_for_cap(max_audio_seconds, target_sample_rate)
    waveform = _resolve_waveform(
        audio_path_or_waveform,
        target_sample_rate=target_sample_rate,
    )
    audio_batch, duration_sec = _prepare_waveform_batch(
        waveform,
        max_samples=max_samples,
        target_sample_rate=target_sample_rate,
    )

    if _model_has_duration_input(model):
        model_input = {
            "audio": audio_batch,
            "duration": np.asarray([duration_sec], dtype=np.float32),
        }
    else:
        model_input = audio_batch

    outputs = model.predict(model_input, verbose=0)
    pred_times, pred_confidence = _extract_prediction_arrays(outputs, model)

    if not _model_has_duration_input(model):
        pred_times = pred_times * np.float32(duration_sec)

    filtered_times, filtered_confidences = _apply_confidence_threshold(
        pred_times,
        pred_confidence,
        confidence_threshold,
    )
    return _apply_min_onset_distance(
        filtered_times,
        filtered_confidences,
        min_onset_distance_ms,
    )
