"""Event-based onset precision, recall, and F1 at a time tolerance."""

import numpy as np
import tensorflow as tf

from stepcovnet.onset_events import matching

DEFAULT_TOLERANCE_SEC = matching.DEFAULT_TOLERANCE_SEC
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MIN_ONSET_DISTANCE_MS = 50.0


def filter_predicted_onsets_numpy(
    times_sec: np.ndarray,
    confidences: np.ndarray,
    confidence_threshold: float,
    min_onset_distance_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep confident predictions and enforce a minimum gap between onset times.

    Matches inference post-processing: threshold, sort by time, drop pairs closer
    than ``min_onset_distance_ms`` (earlier time kept).

    Args:
        times_sec: Predicted onset times in seconds.
        confidences: Prediction confidences in ``[0, 1]``.
        confidence_threshold: Minimum confidence to keep a prediction.
        min_onset_distance_ms: Minimum separation between kept onsets in ms.

    Returns:
        Filtered ``(times_sec, confidences)`` as one-dimensional float32 arrays.

    Raises:
        ValueError: If ``confidence_threshold`` or ``min_onset_distance_ms`` is invalid.
    """
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if min_onset_distance_ms < 0.0:
        raise ValueError("min_onset_distance_ms must be non-negative")

    times = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    keep = conf >= confidence_threshold
    times = times[keep]
    conf = conf[keep]
    if times.size == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    order = np.argsort(times, kind="stable")
    sorted_times = times[order]
    sorted_conf = conf[order]

    min_gap_sec = float(min_onset_distance_ms) / 1000.0
    if min_gap_sec <= 0.0:
        return (
            np.asarray(sorted_times, dtype=np.float32),
            np.asarray(sorted_conf, dtype=np.float32),
        )

    kept_times: list[float] = []
    kept_conf: list[float] = []
    last_kept_time = -np.inf
    for time_sec, confidence in zip(sorted_times, sorted_conf, strict=True):
        if float(time_sec) - last_kept_time >= min_gap_sec:
            kept_times.append(float(time_sec))
            kept_conf.append(float(confidence))
            last_kept_time = float(time_sec)

    return (
        np.asarray(kept_times, dtype=np.float32),
        np.asarray(kept_conf, dtype=np.float32),
    )


def _count_event_onset_errors_single_batch(
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float,
    confidence_threshold: float,
    min_onset_distance_ms: float,
) -> tuple[int, int, int]:
    """Count TP/FP/FN for one batch item."""
    if min_onset_distance_ms > 0.0:
        filtered_times, _filtered_conf = filter_predicted_onsets_numpy(
            pred_times,
            pred_confidence,
            confidence_threshold,
            min_onset_distance_ms,
        )
        if filtered_times.size == 0:
            valid_gt = int(gt_mask.astype(bool).sum())
            return 0, 0, valid_gt

        result = matching.match_onsets_numpy(
            filtered_times.reshape(1, -1),
            gt_times.reshape(1, -1),
            gt_mask.reshape(1, -1),
            tolerance_sec=tolerance_sec,
        )
        num_matches = int(result.num_matches[0])
        num_filtered = filtered_times.size
        false_positives = num_filtered - num_matches
        false_negatives = int(result.gt_unmatched_mask[0].sum())
        return num_matches, false_positives, false_negatives

    result = matching.match_onsets_numpy(
        pred_times.reshape(1, -1),
        gt_times.reshape(1, -1),
        gt_mask.reshape(1, -1),
        tolerance_sec=tolerance_sec,
    )

    conf = pred_confidence.reshape(-1)
    num_matches = int(result.num_matches[0])
    matched_pred = result.matched_pred_indices[0, :num_matches]
    matched_gt = result.matched_gt_indices[0, :num_matches]

    true_positives = 0
    false_negatives = 0
    for pred_idx, _gt_idx in zip(matched_pred, matched_gt, strict=True):
        if conf[pred_idx] >= confidence_threshold:
            true_positives += 1
        else:
            false_negatives += 1

    num_queries = pred_times.size
    pred_unmatched = result.pred_unmatched_mask[0]
    false_positives = 0
    for pred_idx in range(num_queries):
        if pred_unmatched[pred_idx] and conf[pred_idx] >= confidence_threshold:
            false_positives += 1

    false_negatives += int(result.gt_unmatched_mask[0].sum())
    return true_positives, false_positives, false_negatives


def _as_batch_with_confidence(
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Ensure prediction and ground-truth inputs have a leading batch dimension."""
    pred = np.asarray(pred_times, dtype=np.float64)
    confidence = np.asarray(pred_confidence, dtype=np.float64)
    gt = np.asarray(gt_times, dtype=np.float64)
    mask = np.asarray(gt_mask, dtype=np.float64)

    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    if confidence.ndim == 1:
        confidence = confidence[np.newaxis, :]
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    if mask.ndim == 1:
        mask = mask[np.newaxis, :]

    if pred.ndim != 2 or confidence.ndim != 2 or gt.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            "pred_times, pred_confidence, gt_times, and gt_mask must be 1D or 2D arrays"
        )

    batch_size = pred.shape[0]
    if confidence.shape != pred.shape:
        raise ValueError("pred_times and pred_confidence must have the same shape")
    if gt.shape[0] != batch_size or mask.shape != gt.shape:
        raise ValueError(
            "pred_times, gt_times, and gt_mask batch dimensions must match"
        )

    return pred, confidence, gt, mask, batch_size


def count_event_onset_errors_numpy(
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float,
    confidence_threshold: float,
    min_onset_distance_ms: float = 0.0,
) -> tuple[int, int, int]:
    """Count true positives, false positives, and false negatives for one batch.

    When ``min_onset_distance_ms`` is zero, Hungarian matching runs on all ``K``
    query slots. A matched pair counts as a true positive only when the prediction
    confidence is at least ``confidence_threshold``; otherwise that ground-truth
    onset is a false negative. Unmatched query slots with confidence at or above
    the threshold are false positives. Unmatched ground-truth onsets are false
    negatives.

    When ``min_onset_distance_ms`` is positive, predictions are filtered like
    inference (confidence threshold, then minimum gap) before Hungarian matching.
    Surviving predictions that match ground truth within ``tolerance_sec`` are
    true positives; unmatched survivors are false positives.

    Args:
        pred_times: Predicted onset times in seconds; shape ``(num_queries,)`` or
            ``(batch, num_queries)``.
        pred_confidence: Prediction confidences in ``[0, 1]``; same shape as
            ``pred_times``.
        gt_times: Ground-truth onset times in seconds; shape ``(n_max_onsets,)`` or
            ``(batch, n_max_onsets)``.
        gt_mask: Mask marking real GT onsets (1) vs padding (0); same shape as
            ``gt_times``.
        tolerance_sec: Maximum allowed absolute time error for a valid match.
        confidence_threshold: Minimum confidence for a prediction to count.
        min_onset_distance_ms: Minimum gap between kept predictions before
            matching; ``0`` disables inference-style filtering.

    Returns:
        Tuple of ``(true_positives, false_positives, false_negatives)`` summed
        over the batch.
    """
    if min_onset_distance_ms < 0.0:
        raise ValueError("min_onset_distance_ms must be non-negative")

    pred, confidence, gt, mask, batch_size = _as_batch_with_confidence(
        pred_times, pred_confidence, gt_times, gt_mask
    )

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for batch_idx in range(batch_size):
        tp, fp, fn = _count_event_onset_errors_single_batch(
            pred[batch_idx],
            confidence[batch_idx],
            gt[batch_idx],
            mask[batch_idx],
            tolerance_sec,
            confidence_threshold,
            min_onset_distance_ms,
        )
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    return true_positives, false_positives, false_negatives


def _precision_recall_f1_from_counts(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float, float, float]:
    """Convert TP/FP/FN counts to precision, recall, and F1."""
    tp = float(true_positives)
    fp = float(false_positives)
    fn = float(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0
    return precision, recall, f1


def event_onset_f1_numpy(
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_onset_distance_ms: float = 0.0,
) -> tuple[float, float, float]:
    """Compute event onset precision, recall, and F1 in seconds.

    See :func:`count_event_onset_errors_numpy` for matching and filtering rules.

    Args:
        pred_times: Predicted onset times in seconds; shape ``(num_queries,)`` or
            ``(batch, num_queries)``.
        pred_confidence: Prediction confidences in ``[0, 1]``; same shape as
            ``pred_times``.
        gt_times: Ground-truth onset times in seconds; shape ``(n_max_onsets,)`` or
            ``(batch, n_max_onsets)``.
        gt_mask: Mask marking real GT onsets (1) vs padding (0); same shape as
            ``gt_times``.
        tolerance_sec: Maximum allowed absolute time error for a valid match.
        confidence_threshold: Minimum confidence for a prediction to be counted.
        min_onset_distance_ms: Minimum gap before matching; ``0`` uses all slots.

    Returns:
        Tuple of ``(precision, recall, f1)`` aggregated over the batch.
    """
    if tolerance_sec < 0:
        raise ValueError("tolerance_sec must be non-negative")
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if min_onset_distance_ms < 0.0:
        raise ValueError("min_onset_distance_ms must be non-negative")

    counts = count_event_onset_errors_numpy(
        pred_times,
        pred_confidence,
        gt_times,
        gt_mask,
        tolerance_sec,
        confidence_threshold,
        min_onset_distance_ms,
    )
    return _precision_recall_f1_from_counts(*counts)


def _event_onset_f1_numpy_wrapper(
    pred_times: np.ndarray,
    pred_confidence: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: np.ndarray,
    confidence_threshold: np.ndarray,
    min_onset_distance_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy wrapper for ``tf.numpy_function``."""
    precision, recall, f1 = event_onset_f1_numpy(
        pred_times,
        pred_confidence,
        gt_times,
        gt_mask,
        tolerance_sec=float(tolerance_sec.reshape(-1)[0]),
        confidence_threshold=float(confidence_threshold.reshape(-1)[0]),
        min_onset_distance_ms=float(min_onset_distance_ms.reshape(-1)[0]),
    )
    return (
        np.array(precision, dtype=np.float64),
        np.array(recall, dtype=np.float64),
        np.array(f1, dtype=np.float64),
    )


def event_onset_f1(
    pred_times: np.ndarray | tf.Tensor,
    pred_confidence: np.ndarray | tf.Tensor,
    gt_times: np.ndarray | tf.Tensor,
    gt_mask: np.ndarray | tf.Tensor,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_onset_distance_ms: float = 0.0,
) -> tuple[np.ndarray | tf.Tensor, np.ndarray | tf.Tensor, np.ndarray | tf.Tensor]:
    """Compute event onset precision, recall, and F1.

    Accepts NumPy arrays or TensorFlow tensors. Tensor inputs are evaluated via
    ``tf.numpy_function`` wrapping :func:`event_onset_f1_numpy`.

    Args:
        pred_times: Predicted onset times in seconds; shape ``(batch, num_queries)``
            or ``(num_queries,)``.
        pred_confidence: Prediction confidences; same shape as ``pred_times``.
        gt_times: Ground-truth onset times; shape ``(batch, n_max_onsets)`` or
            ``(n_max_onsets,)``.
        gt_mask: Ground-truth validity mask; same shape as ``gt_times``.
        tolerance_sec: Maximum allowed absolute time error for a valid match.
        confidence_threshold: Minimum confidence for a prediction to be counted.
        min_onset_distance_ms: Minimum gap before matching; ``0`` uses all slots.

    Returns:
        Tuple of ``(precision, recall, f1)`` as floats (NumPy scalars or TF
        rank-0 tensors).
    """
    if isinstance(pred_times, tf.Tensor) or isinstance(pred_confidence, tf.Tensor):
        pred_times = tf.cast(tf.convert_to_tensor(pred_times), tf.float32)
        pred_confidence = tf.cast(tf.convert_to_tensor(pred_confidence), tf.float32)
        gt_times = tf.convert_to_tensor(gt_times, dtype=tf.float32)
        gt_mask = tf.convert_to_tensor(gt_mask, dtype=tf.float32)

        outputs = tf.numpy_function(
            _event_onset_f1_numpy_wrapper,
            [
                pred_times,
                pred_confidence,
                gt_times,
                gt_mask,
                np.array([tolerance_sec], dtype=np.float64),
                np.array([confidence_threshold], dtype=np.float64),
                np.array([min_onset_distance_ms], dtype=np.float64),
            ],
            [tf.float64, tf.float64, tf.float64],
        )
        precision, recall, f1 = outputs
        precision = tf.cast(precision, tf.float32)
        recall = tf.cast(recall, tf.float32)
        f1 = tf.cast(f1, tf.float32)
        return precision, recall, f1

    precision, recall, f1 = event_onset_f1_numpy(
        np.asarray(pred_times),
        np.asarray(pred_confidence),
        np.asarray(gt_times),
        np.asarray(gt_mask),
        tolerance_sec=tolerance_sec,
        confidence_threshold=confidence_threshold,
        min_onset_distance_ms=min_onset_distance_ms,
    )
    return (
        np.array(precision, dtype=np.float64),
        np.array(recall, dtype=np.float64),
        np.array(f1, dtype=np.float64),
    )
