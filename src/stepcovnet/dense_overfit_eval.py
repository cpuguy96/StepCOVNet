"""Peak-pick and event-F1 evaluation for dense frame onset models."""

import pathlib

import keras
import numpy as np
import scipy.signal
import tensorflow as tf

from stepcovnet import config, datasets, pairing
from stepcovnet.onset_events import charts, metrics

DEFAULT_MIN_ONSET_DISTANCE_MS = 50.0
DEFAULT_TOLERANCE_SEC = 0.02
DEFAULT_CONFIDENCE_THRESHOLD = 0.05
DENSE_EVENT_ONSET_F1_METRIC_NAME = "dense_event_onset_f1"
DEFAULT_EVENT_F1_SWEEP_THRESHOLDS = (
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
)


def peak_times_and_confidence(
    probabilities: np.ndarray,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    hop_sec: float = datasets.HOP_COEFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Return peak onset times (seconds) and heights from a 1D probability trace."""
    flat = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    min_distance_frames = int(round((min_onset_distance_ms / 1000.0) / hop_sec))
    peak_indices, properties = scipy.signal.find_peaks(
        flat,
        height=confidence_threshold,
        distance=max(1, min_distance_frames),
    )
    if peak_indices.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    times = (peak_indices * hop_sec).astype(np.float32)
    confidences = np.asarray(properties["peak_heights"], dtype=np.float32)
    return times, confidences


def gt_onset_times_from_frame_target(
    target: np.ndarray,
    *,
    hop_sec: float = datasets.HOP_COEFF,
    onset_threshold: float = 0.5,
) -> np.ndarray:
    """Recover sorted onset times in seconds from a dense frame target tensor."""
    frame_values = np.asarray(target, dtype=np.float64)
    if frame_values.ndim >= 2:
        frame_active = frame_values.max(axis=-1) >= onset_threshold
    else:
        frame_active = frame_values.reshape(-1) >= onset_threshold
    frame_indices = np.flatnonzero(frame_active)
    return (frame_indices * hop_sec).astype(np.float32)


def _dense_event_onset_counts_for_sample(
    y_true_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    *,
    tolerance_sec: float,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    hop_sec: float,
) -> tuple[float, float, float]:
    pred_probs = np.asarray(y_pred_sample, dtype=np.float64).reshape(-1)
    pred_times, pred_conf = peak_times_and_confidence(
        pred_probs,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        hop_sec=hop_sec,
    )
    gt_times = gt_onset_times_from_frame_target(y_true_sample, hop_sec=hop_sec)
    n_peaks = int(pred_times.size)
    n_gt = int(gt_times.size)
    n_max = max(n_peaks, n_gt, 1)
    pred_times_batch = np.zeros((1, n_max), dtype=np.float32)
    pred_conf_batch = np.zeros((1, n_max), dtype=np.float32)
    if n_peaks:
        pred_times_batch[0, :n_peaks] = pred_times
        pred_conf_batch[0, :n_peaks] = pred_conf
    gt_times_batch = np.zeros((1, n_max), dtype=np.float32)
    gt_mask_batch = np.zeros((1, n_max), dtype=np.float32)
    if n_gt:
        gt_times_batch[0, :n_gt] = gt_times
        gt_mask_batch[0, :n_gt] = 1.0
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec,
        confidence_threshold,
        min_onset_distance_ms,
    )
    return float(tp), float(fp), float(fn)


def _dense_event_onset_counts_numpy_wrapper(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance_sec: np.ndarray,
    confidence_threshold: np.ndarray,
    min_onset_distance_ms: np.ndarray,
    hop_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_array = np.asarray(y_true, dtype=np.float32)
    y_pred_array = np.asarray(y_pred, dtype=np.float32)
    if y_true_array.ndim == 2:
        y_true_array = y_true_array[:, :, np.newaxis]
    if y_pred_array.ndim == 2:
        y_pred_array = y_pred_array[:, :, np.newaxis]
    batch_size = y_true_array.shape[0]
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    tol = float(tolerance_sec.reshape(-1)[0])
    threshold = float(confidence_threshold.reshape(-1)[0])
    min_gap_ms = float(min_onset_distance_ms.reshape(-1)[0])
    hop = float(hop_sec.reshape(-1)[0])
    for sample_idx in range(batch_size):
        tp, fp, fn = _dense_event_onset_counts_for_sample(
            y_true_array[sample_idx],
            y_pred_array[sample_idx],
            tolerance_sec=tol,
            confidence_threshold=threshold,
            min_onset_distance_ms=min_gap_ms,
            hop_sec=hop,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return (
        np.array(total_tp, dtype=np.float64),
        np.array(total_fp, dtype=np.float64),
        np.array(total_fn, dtype=np.float64),
    )


def dense_event_onset_counts_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tolerance_sec: float,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    hop_sec: float = datasets.HOP_COEFF,
) -> tuple[float, float, float]:
    """Return micro TP/FP/FN for one or more samples in numpy arrays."""
    tp, fp, fn = _dense_event_onset_counts_numpy_wrapper(
        y_true,
        y_pred,
        np.array([tolerance_sec], dtype=np.float64),
        np.array([confidence_threshold], dtype=np.float64),
        np.array([min_onset_distance_ms], dtype=np.float64),
        np.array([hop_sec], dtype=np.float64),
    )
    return float(tp), float(fp), float(fn)


def micro_f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


class DenseValEventF1Callback(keras.callbacks.Callback):
    """Compute peak-pick val event F1 at epoch end (same path as eval script)."""

    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        *,
        confidence_threshold: float,
        tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
        min_onset_distance_ms: float = DEFAULT_MIN_ONSET_DISTANCE_MS,
        metric_name: str = DENSE_EVENT_ONSET_F1_METRIC_NAME,
    ) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.confidence_threshold = confidence_threshold
        self.tolerance_sec = tolerance_sec
        self.min_onset_distance_ms = min_onset_distance_ms
        self.metric_name = metric_name

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        del epoch
        if logs is None:
            logs = {}
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        for features, target in self.val_dataset:
            pred = self.model.predict(features, verbose=0)
            tp, fp, fn = dense_event_onset_counts_from_arrays(
                np.asarray(target),
                np.asarray(pred),
                tolerance_sec=self.tolerance_sec,
                confidence_threshold=self.confidence_threshold,
                min_onset_distance_ms=self.min_onset_distance_ms,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
        logs[f"val_{self.metric_name}"] = micro_f1_from_counts(
            total_tp,
            total_fp,
            total_fn,
        )


def build_gt_batch(
    chart_path: str,
    *,
    n_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded ground-truth batch arrays for event metrics."""
    times = charts.load_onset_times(chart_path, max_steps=None)
    if times is None:
        raise ValueError(f"failed to load chart times: {chart_path}")
    n_gt = int(times.size)
    width = n_max if n_max is not None else max(n_gt, 1)
    if n_gt > width:
        raise ValueError(
            f"chart has {n_gt} steps but n_max={width}: {chart_path}",
        )
    gt_times = np.zeros((1, width), dtype=np.float32)
    gt_mask = np.zeros((1, width), dtype=np.float32)
    gt_times[0, :n_gt] = times.astype(np.float32)
    gt_mask[0, :n_gt] = 1.0
    return gt_times, gt_mask


def _align_event_batches(
    pred_times: np.ndarray,
    pred_conf: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_peaks = int(pred_times.size)
    n_gt = int(gt_mask.sum())
    n_max = max(n_peaks, n_gt, 1)
    pred_times_batch = np.zeros((1, n_max), dtype=np.float32)
    pred_conf_batch = np.zeros((1, n_max), dtype=np.float32)
    if n_peaks:
        pred_times_batch[0, :n_peaks] = pred_times
        pred_conf_batch[0, :n_peaks] = pred_conf
    gt_times_batch = np.zeros((1, n_max), dtype=np.float32)
    gt_mask_batch = np.zeros((1, n_max), dtype=np.float32)
    if n_gt:
        gt_times_batch[0, :n_gt] = gt_times[0, :n_gt]
        gt_mask_batch[0, :n_gt] = gt_mask[0, :n_gt]
    return pred_times_batch, pred_conf_batch, gt_times_batch, gt_mask_batch


def _micro_event_f1(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def _event_metrics_from_batches(
    pred_times_batch: np.ndarray,
    pred_conf_batch: np.ndarray,
    gt_times_batch: np.ndarray,
    gt_mask_batch: np.ndarray,
    *,
    tolerance_sec: float,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    num_peaks: int,
) -> dict[str, float]:
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec,
        confidence_threshold,
        min_onset_distance_ms,
    )
    _p, _r, f1 = metrics.event_onset_f1_numpy(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec,
        confidence_threshold,
        min_onset_distance_ms,
    )
    return {
        "event_f1": float(f1),
        "event_tp": float(tp),
        "event_fp": float(fp),
        "event_fn": float(fn),
        "num_peaks": float(num_peaks),
    }


def predict_dense_probs_for_pair(
    model: tf.keras.Model,
    audio_path: str,
    dataset_config: config.OnsetDatasetConfig,
    *,
    data_root: str = "",
) -> np.ndarray:
    """Load features for one audio pair and return the flat probability trace."""
    features = datasets.load_onset_features(
        audio_path,
        dataset_config.feature_source,
        dataset_config.mert_features_dir,
        data_root or dataset_config.data_dir,
    )
    if config.uses_waveform_model_input(dataset_config):
        features_batch = np.expand_dims(features, axis=0)
    else:
        features = datasets.normalize_onset_spectrogram(features)
        features_batch = np.expand_dims(features, axis=0)
    pred = model.predict(features_batch, verbose=0)
    return np.asarray(pred).reshape(-1)


def event_metrics_from_probs(
    pred_probs: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    tolerance_sec: float,
) -> dict[str, float]:
    """Peak-pick a cached probability trace and score it against ground truth."""
    pred_times, pred_conf = peak_times_and_confidence(
        pred_probs,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
    )
    n_peaks = int(pred_times.size)
    pred_times_batch, pred_conf_batch, gt_times_batch, gt_mask_batch = (
        _align_event_batches(pred_times, pred_conf, gt_times, gt_mask)
    )
    return _event_metrics_from_batches(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec=tolerance_sec,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        num_peaks=n_peaks,
    )


def eval_dense_event_f1_for_pair(
    model: tf.keras.Model,
    audio_path: str,
    chart_path: str,
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    tolerance_sec: float,
    data_root: str = "",
) -> dict[str, float]:
    """Peak-pick event F1 for one audio/chart pair."""
    pred_probs = predict_dense_probs_for_pair(
        model,
        audio_path,
        dataset_config,
        data_root=data_root,
    )
    gt_times, gt_mask = build_gt_batch(chart_path)
    return event_metrics_from_probs(
        pred_probs,
        gt_times,
        gt_mask,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        tolerance_sec=tolerance_sec,
    )


def eval_dense_val_event_f1(
    model: tf.keras.Model,
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float = DEFAULT_MIN_ONSET_DISTANCE_MS,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    val_data_dir: str = "",
) -> dict[str, object]:
    """Peak-pick event F1 on every audio/chart pair under a val directory."""
    val_dir = val_data_dir or dataset_config.val_data_dir
    per_song: dict[str, dict[str, float]] = {}
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    f1_sum = 0.0
    for audio_path, chart_path in pairing.list_audio_chart_pairs(val_dir):
        song = pathlib.Path(audio_path).parent.name
        song_metrics = eval_dense_event_f1_for_pair(
            model,
            audio_path,
            chart_path,
            dataset_config,
            model_config,
            confidence_threshold=confidence_threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            tolerance_sec=tolerance_sec,
            data_root=val_dir,
        )
        per_song[song] = song_metrics
        total_tp += song_metrics["event_tp"]
        total_fp += song_metrics["event_fp"]
        total_fn += song_metrics["event_fn"]
        f1_sum += song_metrics["event_f1"]
    n_songs = len(per_song)
    mean_f1 = float(f1_sum / n_songs) if n_songs else 0.0
    micro_f1 = _micro_event_f1(total_tp, total_fp, total_fn)
    micro_p_denom = total_tp + total_fp
    micro_r_denom = total_tp + total_fn
    return {
        "eval_split": val_dir,
        "num_songs": n_songs,
        "mean_event_f1": mean_f1,
        "micro_event_f1": micro_f1,
        "micro_precision": float(total_tp / micro_p_denom) if micro_p_denom else 0.0,
        "micro_recall": float(total_tp / micro_r_denom) if micro_r_denom else 0.0,
        "micro_tp": total_tp,
        "micro_fp": total_fp,
        "micro_fn": total_fn,
        "eval_kwargs": {
            "confidence_threshold": confidence_threshold,
            "min_onset_distance_ms": min_onset_distance_ms,
            "tolerance_sec": tolerance_sec,
        },
        "per_song": per_song,
    }


def _threshold_summary_from_cache(
    pred_gt_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    confidence_threshold: float,
    *,
    min_onset_distance_ms: float,
    tolerance_sec: float,
) -> dict[str, float]:
    """Aggregate micro/mean event F1 over cached predictions at one threshold."""
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    f1_sum = 0.0
    for pred_probs, gt_times, gt_mask in pred_gt_cache:
        song_metrics = event_metrics_from_probs(
            pred_probs,
            gt_times,
            gt_mask,
            confidence_threshold=confidence_threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            tolerance_sec=tolerance_sec,
        )
        total_tp += song_metrics["event_tp"]
        total_fp += song_metrics["event_fp"]
        total_fn += song_metrics["event_fn"]
        f1_sum += song_metrics["event_f1"]
    n_songs = len(pred_gt_cache)
    return {
        "confidence_threshold": confidence_threshold,
        "micro_event_f1": _micro_event_f1(total_tp, total_fp, total_fn),
        "mean_event_f1": float(f1_sum / n_songs) if n_songs else 0.0,
        "micro_tp": total_tp,
        "micro_fp": total_fp,
        "micro_fn": total_fn,
    }


def sweep_thresholds_dense_val_event_f1(
    model: tf.keras.Model,
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    *,
    thresholds: tuple[float, ...] = DEFAULT_EVENT_F1_SWEEP_THRESHOLDS,
    min_onset_distance_ms: float = DEFAULT_MIN_ONSET_DISTANCE_MS,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    val_data_dir: str = "",
) -> dict[str, object]:
    """Predict once per val pair, then score event F1 across confidence thresholds.

    The model forward pass runs a single time per audio/chart pair; the cached
    probability traces are re-scored at every threshold so the sweep cost is
    dominated by one inference pass over the validation split.
    """
    del model_config
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    val_dir = val_data_dir or dataset_config.val_data_dir
    pred_gt_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for audio_path, chart_path in pairing.list_audio_chart_pairs(val_dir):
        pred_probs = predict_dense_probs_for_pair(
            model,
            audio_path,
            dataset_config,
            data_root=val_dir,
        )
        gt_times, gt_mask = build_gt_batch(chart_path)
        pred_gt_cache.append((pred_probs, gt_times, gt_mask))

    per_threshold: list[dict[str, float]] = []
    best_summary: dict[str, float] | None = None
    for threshold in thresholds:
        summary = _threshold_summary_from_cache(
            pred_gt_cache,
            threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            tolerance_sec=tolerance_sec,
        )
        per_threshold.append(summary)
        if (
            best_summary is None
            or summary["micro_event_f1"] > best_summary["micro_event_f1"]
        ):
            best_summary = summary
    assert best_summary is not None
    return {
        "eval_split": val_dir,
        "num_songs": len(pred_gt_cache),
        "best_threshold": best_summary["confidence_threshold"],
        "best_micro_event_f1": best_summary["micro_event_f1"],
        "best_mean_event_f1": best_summary["mean_event_f1"],
        "per_threshold": per_threshold,
    }


def eval_dense_event_f1(
    model: tf.keras.Model,
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    chart_path: str,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    tolerance_sec: float,
) -> dict[str, float]:
    """Run dense forward pass, peak-pick, and event F1 vs chart ground truth."""
    ds = datasets.create_dataset(
        dataset_config.val_data_dir,
        batch_size=1,
        apply_temporal_augment=False,
        should_apply_spec_augment=False,
        use_gaussian_target=False,
        feature_source=dataset_config.feature_source,
        mert_features_dir=dataset_config.mert_features_dir,
        n_features=config.resolve_onset_input_features(
            dataset_config,
            model_config,
        ),
    )
    features, _target = next(iter(ds.take(1)))
    pred = model.predict(features, verbose=0)
    pred_probs = np.asarray(pred).reshape(-1)
    pred_times, pred_conf = peak_times_and_confidence(
        pred_probs,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
    )
    gt_times, gt_mask = build_gt_batch(chart_path)
    n_peaks = int(pred_times.size)
    pred_times_batch, pred_conf_batch, gt_times_batch, gt_mask_batch = (
        _align_event_batches(pred_times, pred_conf, gt_times, gt_mask)
    )
    return _event_metrics_from_batches(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec=tolerance_sec,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        num_peaks=n_peaks,
    )
