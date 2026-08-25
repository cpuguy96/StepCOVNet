"""Peak-pick and event-F1 evaluation for dense frame onset models."""

import keras
import numpy as np
import scipy.signal
import tensorflow as tf

from stepcovnet import config, datasets, onset_null_baseline, timing_match
from stepcovnet.onset_events import charts, metrics

DEFAULT_MIN_ONSET_DISTANCE_MS = 50.0
DEFAULT_TOLERANCE_SEC = timing_match.DEFAULT_TOLERANCE_SEC
DEFAULT_CONFIDENCE_THRESHOLD = 0.05
CORRUPTION_KINDS = ("none", "shuffle", "zeros")
DEFAULT_CORRUPTION_SEED = 42
DENSE_EVENT_ONSET_F1_METRIC_NAME = "dense_event_onset_f1"
DENSE_SKILL_EVENT_F1_METRIC_NAME = "skill_event_f1"
DENSE_TIMING_MATCH_METRIC_NAME = timing_match.TIMING_MATCH_METRIC_NAME
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


def _dense_timing_match_for_sample(
    y_true_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    *,
    tolerance_sec: float,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    hop_sec: float,
) -> tuple[float, float, float]:
    pred_probs = np.asarray(y_pred_sample, dtype=np.float64).reshape(-1)
    pred_times, _pred_conf = peak_times_and_confidence(
        pred_probs,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        hop_sec=hop_sec,
    )
    ref_times = gt_onset_times_from_frame_target(y_true_sample, hop_sec=hop_sec)
    n_matched, n_ref = timing_match.timing_match_counts_numpy(
        pred_times,
        ref_times,
        tolerance_sec=tolerance_sec,
    )
    return float(n_matched), float(n_ref), float(pred_times.size)


def dense_timing_match_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tolerance_sec: float,
    confidence_threshold: float,
    min_onset_distance_ms: float,
    hop_sec: float = datasets.HOP_COEFF,
) -> tuple[float, float, float]:
    """Return micro (n_matched, n_ref, n_pred) for dense peak-pick vs frame GT."""
    y_true_array = np.asarray(y_true, dtype=np.float32)
    y_pred_array = np.asarray(y_pred, dtype=np.float32)
    if y_true_array.ndim == 2:
        y_true_array = y_true_array[:, :, np.newaxis]
    if y_pred_array.ndim == 2:
        y_pred_array = y_pred_array[:, :, np.newaxis]
    total_matched = 0.0
    total_ref = 0.0
    total_pred = 0.0
    for sample_idx in range(y_true_array.shape[0]):
        n_matched, n_ref, n_pred = _dense_timing_match_for_sample(
            y_true_array[sample_idx],
            y_pred_array[sample_idx],
            tolerance_sec=tolerance_sec,
            confidence_threshold=confidence_threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            hop_sec=hop_sec,
        )
        total_matched += n_matched
        total_ref += n_ref
        total_pred += n_pred
    return total_matched, total_ref, total_pred


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
    """Compute peak-pick val event metrics at epoch end (same path as eval script).

    Both published metrics depend on the peak-pick threshold, so the threshold
    is swept each epoch and the best operating point is reported. Selection
    ranks on skill over the audio-blind floor rather than raw F1: lowering the
    threshold emits more peaks, which raises the chance floor along with the
    raw score, so a sweep ranked on raw F1 can pick a threshold that only
    matched the onset *count*.

    Ordered ``timing_match`` is published as a diagnostic but is not used for
    selection on multi-song dense val. It requires the ``i``th prediction to
    align with the ``i``th reference, so a single inserted or dropped onset
    desynchronizes everything after it; it stays pinned near its ~0 floor
    across the whole range of partial quality and cannot rank checkpoints.
    """

    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        *,
        confidence_threshold: float,
        tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
        min_onset_distance_ms: float = DEFAULT_MIN_ONSET_DISTANCE_MS,
        metric_name: str = DENSE_EVENT_ONSET_F1_METRIC_NAME,
        sweep_thresholds: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.confidence_threshold = confidence_threshold
        self.tolerance_sec = tolerance_sec
        self.min_onset_distance_ms = min_onset_distance_ms
        self.metric_name = metric_name
        candidates = sweep_thresholds or DEFAULT_EVENT_F1_SWEEP_THRESHOLDS
        self.sweep_thresholds = tuple(
            sorted({float(confidence_threshold), *(float(t) for t in candidates)}),
        )

    def _summary_at(
        self,
        cached: list[tuple[np.ndarray, np.ndarray]],
        threshold: float,
    ) -> dict[str, float]:
        total_tp = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_matched = 0.0
        total_ref = 0.0
        total_pred = 0.0
        null_rows: list[list[onset_null_baseline.NullCounts]] = []
        for target_arr, pred_arr in cached:
            tp, fp, fn = dense_event_onset_counts_from_arrays(
                target_arr,
                pred_arr,
                tolerance_sec=self.tolerance_sec,
                confidence_threshold=threshold,
                min_onset_distance_ms=self.min_onset_distance_ms,
            )
            n_matched, n_ref, n_pred = dense_timing_match_from_arrays(
                target_arr,
                pred_arr,
                tolerance_sec=self.tolerance_sec,
                confidence_threshold=threshold,
                min_onset_distance_ms=self.min_onset_distance_ms,
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_matched += n_matched
            total_ref += n_ref
            total_pred += n_pred
            ref_times = gt_onset_times_from_frame_target(target_arr)
            null_rows.append(
                onset_null_baseline.null_counts_for_song(
                    np.asarray(ref_times, dtype=np.float64),
                    duration_sec=float(
                        np.asarray(pred_arr).reshape(-1).size * datasets.HOP_COEFF,
                    ),
                    n_pred=int(n_pred),
                    tolerance_sec=self.tolerance_sec,
                    hop_sec=float(datasets.HOP_COEFF),
                ),
            )
        micro_f1 = micro_f1_from_counts(total_tp, total_fp, total_fn)
        micro_tm = timing_match.micro_timing_match_rate(
            total_matched,
            total_ref,
            total_pred,
        )
        _kind, f1_floor = onset_null_baseline.strongest_null(
            onset_null_baseline.aggregate_null_counts(null_rows),
            metric="event_f1",
        )
        return {
            "event_f1": micro_f1,
            "timing_match": micro_tm,
            "null_event_f1_floor": f1_floor,
            "skill_event_f1": onset_null_baseline.skill_over_null(micro_f1, f1_floor),
        }

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        del epoch
        if logs is None:
            logs = {}
        cached = [
            (np.asarray(target), np.asarray(self.model.predict(features, verbose=0)))
            for features, target in self.val_dataset
        ]

        best: dict[str, float] | None = None
        best_threshold = self.confidence_threshold
        for threshold in self.sweep_thresholds:
            summary = self._summary_at(cached, threshold)
            if best is None or summary["skill_event_f1"] > best["skill_event_f1"]:
                best = summary
                best_threshold = threshold
        if best is None:
            return

        logs[f"val_{self.metric_name}"] = best["event_f1"]
        logs[f"val_{DENSE_SKILL_EVENT_F1_METRIC_NAME}"] = best["skill_event_f1"]
        logs[f"val_{DENSE_SKILL_EVENT_F1_METRIC_NAME}_floor"] = best[
            "null_event_f1_floor"
        ]
        logs[f"val_{DENSE_TIMING_MATCH_METRIC_NAME}"] = best["timing_match"]
        logs[f"val_{DENSE_TIMING_MATCH_METRIC_NAME}_threshold"] = best_threshold


def build_gt_batch(
    chart_path: str,
    *,
    n_max: int | None = None,
    chart_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded ground-truth batch arrays for event metrics."""
    times = charts.load_onset_times(
        chart_path,
        max_steps=None,
        chart_index=chart_index,
    )
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
    flat_pred_times: np.ndarray | None = None,
    flat_ref_times: np.ndarray | None = None,
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
    result: dict[str, float] = {
        "event_f1": float(f1),
        "event_tp": float(tp),
        "event_fp": float(fp),
        "event_fn": float(fn),
        "num_peaks": float(num_peaks),
    }
    if flat_pred_times is not None and flat_ref_times is not None:
        timing = timing_match.timing_match_report(
            flat_pred_times,
            flat_ref_times,
            tolerance_sec=tolerance_sec,
        )
        result["timing_match_n_matched"] = float(timing["n_matched"])
        result["timing_match_n_pred"] = float(timing["n_pred"])
        result["timing_match_n_ref"] = float(timing["n_ref"])
        result["timing_match_n_denom"] = float(timing["n_denom"])
        result["timing_match_rate"] = float(timing["rate"])
    return result


def corrupt_features(
    features: np.ndarray,
    corruption: str,
    *,
    seed: int = DEFAULT_CORRUPTION_SEED,
) -> np.ndarray:
    """Return audio features corrupted for a grounding ablation.

    A dense model whose peak-pick output survives ``shuffle`` or ``zeros``
    unchanged is not reading the audio; scores under corruption should
    collapse toward the audio-blind null floor.

    Args:
        features: Feature array with time on axis 0.
        corruption: One of :data:`CORRUPTION_KINDS` — ``none`` returns the
            input untouched, ``shuffle`` permutes frames along time with a
            seeded generator, ``zeros`` replaces all values with zeros.
        seed: Generator seed for ``shuffle`` so ablations are reproducible.

    Returns:
        Corrupted copy (or the original array for ``none``).

    Raises:
        ValueError: If ``corruption`` is not a supported kind.
    """
    if corruption == "none":
        return features
    if corruption == "zeros":
        return np.zeros_like(features)
    if corruption == "shuffle":
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(features.shape[0])
        return np.asarray(features)[permutation]
    raise ValueError(f"unsupported corruption kind: {corruption!r}")


def oracle_density_for_pair(
    chart_path: str,
    *,
    duration_sec: float,
    chart_index: int = 0,
    dataset_config: config.OnsetDatasetConfig,
) -> float | None:
    """Return chart-oracle density when conditioning is enabled, else ``None``."""
    if dataset_config.density_conditioning == config.DENSITY_CONDITIONING_NONE:
        return None
    ref_times = charts.load_onset_times(
        chart_path,
        max_steps=None,
        chart_index=chart_index,
    )
    return datasets.onset_density_scalar(int(ref_times.size), duration_sec)


def predict_dense_probs_for_pair(
    model: tf.keras.Model,
    audio_path: str,
    dataset_config: config.OnsetDatasetConfig,
    *,
    data_root: str = "",
    corruption: str = "none",
    density: float | None = None,
) -> np.ndarray:
    """Load features for one audio pair and return the flat probability trace.

    Args:
        model: Trained dense onset model.
        audio_path: Audio file the features belong to.
        dataset_config: Dataset settings (feature source, feature dirs).
        data_root: Root for nested feature paths; falls back to config dirs.
        corruption: Optional grounding ablation from :data:`CORRUPTION_KINDS`
            applied to the features before the forward pass.
        density: Density conditioning value, required when the dataset config
            enables it. Deriving this from the chart being scored is an oracle
            signal — it tells the model how many onsets to emit — so report any
            such run as density-conditioned rather than blind.

    Returns:
        Flat per-frame onset probability trace.

    Raises:
        ValueError: If density conditioning is enabled but ``density`` is None.
    """
    features = datasets.load_onset_features(
        audio_path,
        dataset_config.feature_source,
        dataset_config.mert_features_dir,
        data_root or str(dataset_config.data_root).strip() or dataset_config.data_dir,
    )
    features = corrupt_features(features, corruption)
    if config.uses_waveform_model_input(dataset_config):
        features_batch = np.expand_dims(features, axis=0)
    else:
        features = datasets.normalize_onset_spectrogram(features)
        if dataset_config.density_conditioning != config.DENSITY_CONDITIONING_NONE:
            if density is None:
                raise ValueError(
                    "density is required when density_conditioning is enabled; "
                    "pass the target rate the model should be asked to produce"
                )
            features = datasets.append_density_channel(features, density)
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
    ref_times = timing_match.reference_times_from_mask(gt_times[0], gt_mask[0])
    return _event_metrics_from_batches(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec=tolerance_sec,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        num_peaks=n_peaks,
        flat_pred_times=pred_times,
        flat_ref_times=ref_times,
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
    chart_index: int = 0,
    corruption: str = "none",
) -> dict[str, float]:
    """Peak-pick event F1 for one audio/chart pair.

    Args:
        model: Trained dense onset model.
        audio_path: Audio file for the pair.
        chart_path: Chart file with ground-truth onsets.
        dataset_config: Dataset settings (feature source, feature dirs).
        model_config: Unused; kept for signature parity.
        confidence_threshold: Peak-pick height threshold.
        min_onset_distance_ms: Peak-pick minimum onset spacing.
        tolerance_sec: Match tolerance for event F1 and ``timing_match``.
        data_root: Root for nested feature paths.
        chart_index: Block index when ``chart_path`` is ``.chart.json``.
        corruption: Optional grounding ablation from :data:`CORRUPTION_KINDS`.

    Returns:
        Per-pair metrics including ``event_f1``, ``timing_match_*`` counts,
        ``num_peaks``, and ``duration_sec``.
    """
    del model_config
    features = datasets.load_onset_features(
        audio_path,
        dataset_config.feature_source,
        dataset_config.mert_features_dir,
        data_root or str(dataset_config.data_root).strip() or dataset_config.data_dir,
    )
    duration_sec = float(features.shape[0] * datasets.HOP_COEFF)
    density = oracle_density_for_pair(
        chart_path,
        duration_sec=duration_sec,
        chart_index=chart_index,
        dataset_config=dataset_config,
    )
    pred_probs = predict_dense_probs_for_pair(
        model,
        audio_path,
        dataset_config,
        data_root=data_root,
        corruption=corruption,
        density=density,
    )
    gt_times, gt_mask = build_gt_batch(chart_path, chart_index=chart_index)
    result = event_metrics_from_probs(
        pred_probs,
        gt_times,
        gt_mask,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        tolerance_sec=tolerance_sec,
    )
    result["duration_sec"] = float(pred_probs.size * datasets.HOP_COEFF)
    return result


def eval_dense_val_event_f1(
    model: tf.keras.Model,
    dataset_config: config.OnsetDatasetConfig,
    model_config: config.OnsetModelConfig,
    *,
    confidence_threshold: float,
    min_onset_distance_ms: float = DEFAULT_MIN_ONSET_DISTANCE_MS,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    val_data_dir: str = "",
    corruption: str = "none",
) -> dict[str, object]:
    """Peak-pick event F1 on every val sample (manifest rows or legacy pairs).

    Args:
        model: Trained dense onset model.
        dataset_config: Dataset settings used to resolve val samples/features.
        model_config: Model settings (unused beyond pair eval signature).
        confidence_threshold: Peak-pick height threshold.
        min_onset_distance_ms: Peak-pick minimum onset spacing.
        tolerance_sec: Match tolerance for event F1 and ``timing_match``.
        val_data_dir: Optional explicit val split reference.
        corruption: Optional grounding ablation from :data:`CORRUPTION_KINDS`
            applied to features before every forward pass.

    Returns:
        Aggregate report including per-song metrics and audio-blind
        ``null_floors`` computed at the model's own prediction counts.
    """
    samples, data_root = datasets.resolve_dense_eval_samples(
        dataset_config,
        data_ref=val_data_dir,
        split="val",
    )
    per_sample: dict[str, dict[str, float]] = {}
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    f1_sum = 0.0
    total_matched = 0.0
    total_ref = 0.0
    total_pred = 0.0
    null_rows: list[list[onset_null_baseline.NullCounts]] = []
    for audio_path, chart_path, chart_index in samples:
        sample_key = datasets.dense_eval_sample_key(
            audio_path,
            chart_path,
            chart_index,
        )
        sample_metrics = eval_dense_event_f1_for_pair(
            model,
            audio_path,
            chart_path,
            dataset_config,
            model_config,
            confidence_threshold=confidence_threshold,
            min_onset_distance_ms=min_onset_distance_ms,
            tolerance_sec=tolerance_sec,
            data_root=data_root,
            chart_index=chart_index,
            corruption=corruption,
        )
        per_sample[sample_key] = sample_metrics
        total_tp += sample_metrics["event_tp"]
        total_fp += sample_metrics["event_fp"]
        total_fn += sample_metrics["event_fn"]
        f1_sum += sample_metrics["event_f1"]
        total_matched += sample_metrics.get("timing_match_n_matched", 0.0)
        total_ref += sample_metrics.get("timing_match_n_ref", 0.0)
        total_pred += sample_metrics.get("timing_match_n_pred", 0.0)
        gt_flat = charts.load_onset_times(
            chart_path,
            max_steps=None,
            chart_index=chart_index,
        )
        if gt_flat is not None:
            null_rows.append(
                onset_null_baseline.null_counts_for_song(
                    np.asarray(gt_flat, dtype=np.float64),
                    duration_sec=float(sample_metrics.get("duration_sec", 0.0)),
                    n_pred=int(sample_metrics.get("num_peaks", 0.0)),
                    tolerance_sec=tolerance_sec,
                    hop_sec=float(datasets.HOP_COEFF),
                ),
            )
    n_samples = len(per_sample)
    mean_f1 = float(f1_sum / n_samples) if n_samples else 0.0
    micro_f1 = _micro_event_f1(total_tp, total_fp, total_fn)
    micro_timing_match = timing_match.micro_timing_match_rate(
        total_matched,
        total_ref,
        total_pred,
    )
    micro_p_denom = total_tp + total_fp
    micro_r_denom = total_tp + total_fn
    null_report: dict[str, object] = {}
    if null_rows:
        aggregated = onset_null_baseline.aggregate_null_counts(null_rows)
        f1_kind, f1_floor = onset_null_baseline.strongest_null(
            aggregated,
            metric="event_f1",
        )
        tm_kind, tm_floor = onset_null_baseline.strongest_null(
            aggregated,
            metric="timing_match",
        )
        null_report = {
            "by_kind": aggregated,
            "strongest": {"event_f1": f1_kind, "timing_match": tm_kind},
            "event_f1_floor": f1_floor,
            "timing_match_floor": tm_floor,
            "skill_event_f1": onset_null_baseline.skill_over_null(
                micro_f1,
                f1_floor,
            ),
            "skill_timing_match": onset_null_baseline.skill_over_null(
                micro_timing_match,
                tm_floor,
            ),
        }
    return {
        "eval_split": val_data_dir
        or dataset_config.training_index_path
        or dataset_config.val_data_dir,
        "corruption": corruption,
        "null_floors": null_report,
        "num_songs": n_samples,
        "mean_event_f1": mean_f1,
        "micro_event_f1": micro_f1,
        "micro_timing_match": micro_timing_match,
        "timing_match_n_matched": total_matched,
        "timing_match_n_pred": total_pred,
        "timing_match_n_ref": total_ref,
        "timing_match_n_denom": float(
            timing_match.timing_match_denom(int(total_pred), int(total_ref)),
        ),
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
        "per_song": per_sample,
    }


def _threshold_summary_from_cache(
    pred_gt_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    confidence_threshold: float,
    *,
    min_onset_distance_ms: float,
    tolerance_sec: float,
) -> dict[str, float]:
    """Aggregate micro event F1, ``timing_match``, and null-relative skill.

    Lowering the threshold emits more peaks, which raises the audio-blind floor
    as well as the raw score, so a sweep ranked on raw F1 can select a threshold
    that only matched the onset *count*. Each threshold is therefore scored
    against a null baseline drawn at that threshold's own prediction count.

    Args:
        pred_gt_cache: Per-song ``(pred_probs, gt_times, gt_mask)`` triples.
        confidence_threshold: Peak-pick height threshold to score.
        min_onset_distance_ms: Peak-pick minimum onset spacing.
        tolerance_sec: Match tolerance.

    Returns:
        Micro counts, ``micro_event_f1``, ``micro_timing_match``, the null
        floors at this prediction count, and skill over those floors.
    """
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    f1_sum = 0.0
    total_matched = 0.0
    total_ref = 0.0
    total_pred = 0.0
    null_rows: list[list[onset_null_baseline.NullCounts]] = []
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
        total_matched += song_metrics.get("timing_match_n_matched", 0.0)
        total_ref += song_metrics.get("timing_match_n_ref", 0.0)
        total_pred += song_metrics.get("timing_match_n_pred", 0.0)
        ref_times = timing_match.reference_times_from_mask(gt_times[0], gt_mask[0])
        null_rows.append(
            onset_null_baseline.null_counts_for_song(
                np.asarray(ref_times, dtype=np.float64),
                duration_sec=float(pred_probs.size * datasets.HOP_COEFF),
                n_pred=int(song_metrics["num_peaks"]),
                tolerance_sec=tolerance_sec,
                hop_sec=float(datasets.HOP_COEFF),
            ),
        )
    n_songs = len(pred_gt_cache)
    micro_f1 = _micro_event_f1(total_tp, total_fp, total_fn)
    micro_tm = timing_match.micro_timing_match_rate(
        total_matched,
        total_ref,
        total_pred,
    )
    aggregated = onset_null_baseline.aggregate_null_counts(null_rows)
    _f1_kind, f1_floor = onset_null_baseline.strongest_null(
        aggregated,
        metric="event_f1",
    )
    _tm_kind, tm_floor = onset_null_baseline.strongest_null(
        aggregated,
        metric="timing_match",
    )
    return {
        "confidence_threshold": confidence_threshold,
        "micro_event_f1": micro_f1,
        "mean_event_f1": float(f1_sum / n_songs) if n_songs else 0.0,
        "micro_timing_match": micro_tm,
        "micro_tp": total_tp,
        "micro_fp": total_fp,
        "micro_fn": total_fn,
        "num_peaks": total_pred,
        "null_event_f1_floor": f1_floor,
        "null_timing_match_floor": tm_floor,
        "skill_event_f1": onset_null_baseline.skill_over_null(micro_f1, f1_floor),
        "skill_timing_match": onset_null_baseline.skill_over_null(micro_tm, tm_floor),
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
    selection_metric: str = "skill_event_f1",
) -> dict[str, object]:
    """Predict once per val pair, then score event F1 across confidence thresholds.

    The model forward pass runs a single time per audio/chart pair; the cached
    probability traces are re-scored at every threshold so the sweep cost is
    dominated by one inference pass over the validation split.

    Args:
        model: Trained dense onset model.
        dataset_config: Dataset settings used to resolve val samples/features.
        model_config: Unused; kept for signature parity.
        thresholds: Peak-pick confidence thresholds to score.
        min_onset_distance_ms: Peak-pick minimum onset spacing.
        tolerance_sec: Match tolerance.
        val_data_dir: Optional explicit val split reference.
        selection_metric: Key from the per-threshold summary used to pick the
            best threshold. Defaults to skill over the audio-blind floor on
            Hungarian F1; ranking on raw ``micro_event_f1`` can select a
            threshold that merely matched the onset count. Ordered
            ``timing_match`` stays a diagnostic — it sits at its chance floor
            for a partially-correct model.

    Returns:
        Sweep report with per-threshold summaries and the selected threshold.

    Raises:
        ValueError: If ``thresholds`` is empty.
    """
    del model_config
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    samples, data_root = datasets.resolve_dense_eval_samples(
        dataset_config,
        data_ref=val_data_dir,
        split="val",
    )
    pred_gt_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for audio_path, chart_path, chart_index in samples:
        density = None
        if dataset_config.density_conditioning != config.DENSITY_CONDITIONING_NONE:
            features = datasets.load_onset_features(
                audio_path,
                dataset_config.feature_source,
                dataset_config.mert_features_dir,
                data_root,
            )
            density = oracle_density_for_pair(
                chart_path,
                duration_sec=float(features.shape[0] * datasets.HOP_COEFF),
                chart_index=chart_index,
                dataset_config=dataset_config,
            )
        pred_probs = predict_dense_probs_for_pair(
            model,
            audio_path,
            dataset_config,
            data_root=data_root,
            density=density,
        )
        gt_times, gt_mask = build_gt_batch(chart_path, chart_index=chart_index)
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
        if best_summary is None or (
            summary[selection_metric] > best_summary[selection_metric]
        ):
            best_summary = summary
    assert best_summary is not None
    best_raw_f1 = max(per_threshold, key=lambda row: row["micro_event_f1"])
    eval_split = (
        val_data_dir
        or dataset_config.training_index_path
        or dataset_config.val_data_dir
    )
    return {
        "eval_split": eval_split,
        "num_songs": len(pred_gt_cache),
        "selection_metric": selection_metric,
        "best_threshold": best_summary["confidence_threshold"],
        "best_micro_event_f1": best_summary["micro_event_f1"],
        "best_mean_event_f1": best_summary["mean_event_f1"],
        "best_micro_timing_match": best_summary["micro_timing_match"],
        "best_skill_event_f1": best_summary["skill_event_f1"],
        "best_skill_timing_match": best_summary["skill_timing_match"],
        # Kept so a null-aware selection can be compared against the raw-F1
        # choice that the AR density-conditioning result showed to be unsafe.
        "raw_f1_best_threshold": best_raw_f1["confidence_threshold"],
        "raw_f1_best_micro_event_f1": best_raw_f1["micro_event_f1"],
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
    ref_times = timing_match.reference_times_from_mask(gt_times[0], gt_mask[0])
    return _event_metrics_from_batches(
        pred_times_batch,
        pred_conf_batch,
        gt_times_batch,
        gt_mask_batch,
        tolerance_sec=tolerance_sec,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
        num_peaks=n_peaks,
        flat_pred_times=pred_times,
        flat_ref_times=ref_times,
    )
