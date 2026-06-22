"""Hungarian assignment between prediction slots and ground-truth onsets."""

import dataclasses

import numpy as np
import scipy.optimize
import tensorflow as tf

DEFAULT_TOLERANCE_SEC = 0.02

_LARGE_COST = 1e6


@dataclasses.dataclass
class MatchResult:
    """Hungarian match between prediction slots and ground-truth onsets.

    Attributes:
        matched_pred_indices: Matched prediction slot indices per batch item;
            shape ``(batch, max_matches)`` with unused entries set to ``-1``.
        matched_gt_indices: Matched ground-truth slot indices per batch item;
            same shape as ``matched_pred_indices``.
        num_matches: Count of valid matches per batch item; shape ``(batch,)``.
        pred_unmatched_mask: True where a prediction slot has no GT match;
            shape ``(batch, num_queries)``.
        gt_unmatched_mask: True where a valid GT onset has no prediction match;
            shape ``(batch, n_max_onsets)``; padded GT positions are always False.
    """

    matched_pred_indices: np.ndarray
    matched_gt_indices: np.ndarray
    num_matches: np.ndarray
    pred_unmatched_mask: np.ndarray
    gt_unmatched_mask: np.ndarray


def _as_batch(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Ensure inputs have a leading batch dimension."""
    pred = np.asarray(pred_times, dtype=np.float64)
    gt = np.asarray(gt_times, dtype=np.float64)
    mask = np.asarray(gt_mask, dtype=np.float64)

    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    if mask.ndim == 1:
        mask = mask[np.newaxis, :]

    if pred.ndim != 2 or gt.ndim != 2 or mask.ndim != 2:
        raise ValueError("pred_times, gt_times, and gt_mask must be 1D or 2D arrays")

    batch_size, num_queries = pred.shape
    _, n_max = gt.shape
    if gt.shape[0] != batch_size or mask.shape != gt.shape:
        raise ValueError(
            "pred_times, gt_times, and gt_mask batch dimensions must match"
        )

    return pred, gt, mask, batch_size, num_queries, n_max


def _assign_single_l1(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Run Hungarian assignment on raw L1 cost for training-time pairing."""
    num_queries = pred_times.shape[0]
    n_max = gt_times.shape[0]
    max_matches = min(num_queries, n_max)

    valid_gt = gt_mask.astype(bool)
    gt_slot_indices = np.flatnonzero(valid_gt)
    num_valid_gt = gt_slot_indices.size

    matched_pred = np.full(max_matches, -1, dtype=np.int32)
    matched_gt = np.full(max_matches, -1, dtype=np.int32)
    pred_unmatched = np.ones(num_queries, dtype=bool)
    gt_unmatched = np.zeros(n_max, dtype=bool)

    if num_valid_gt == 0:
        return matched_pred, matched_gt, 0, pred_unmatched, gt_unmatched

    gt_values = gt_times[gt_slot_indices]
    diff = np.abs(pred_times[:, np.newaxis] - gt_values[np.newaxis, :])

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(diff)

    gt_unmatched[valid_gt] = True

    match_count = 0
    for row, col in zip(row_ind, col_ind, strict=False):
        gt_slot = int(gt_slot_indices[col])
        matched_pred[match_count] = int(row)
        matched_gt[match_count] = gt_slot
        pred_unmatched[row] = False
        gt_unmatched[gt_slot] = False
        match_count += 1

    return matched_pred, matched_gt, match_count, pred_unmatched, gt_unmatched


def _match_single(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Run Hungarian matching for one batch item."""
    num_queries = pred_times.shape[0]
    n_max = gt_times.shape[0]
    max_matches = min(num_queries, n_max)

    valid_gt = gt_mask.astype(bool)
    gt_slot_indices = np.flatnonzero(valid_gt)
    num_valid_gt = gt_slot_indices.size

    matched_pred = np.full(max_matches, -1, dtype=np.int32)
    matched_gt = np.full(max_matches, -1, dtype=np.int32)
    pred_unmatched = np.ones(num_queries, dtype=bool)
    gt_unmatched = np.zeros(n_max, dtype=bool)

    if num_valid_gt == 0:
        return matched_pred, matched_gt, 0, pred_unmatched, gt_unmatched

    gt_values = gt_times[gt_slot_indices]
    diff = np.abs(pred_times[:, np.newaxis] - gt_values[np.newaxis, :])
    cost = np.where(diff <= tolerance_sec, diff, _LARGE_COST)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)

    gt_unmatched[valid_gt] = True

    match_count = 0
    for row, col in zip(row_ind, col_ind, strict=False):
        if diff[row, col] <= tolerance_sec:
            gt_slot = int(gt_slot_indices[col])
            matched_pred[match_count] = int(row)
            matched_gt[match_count] = gt_slot
            pred_unmatched[row] = False
            gt_unmatched[gt_slot] = False
            match_count += 1

    return matched_pred, matched_gt, match_count, pred_unmatched, gt_unmatched


def match_onsets_numpy(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    pred_confidence: np.ndarray | None = None,
) -> MatchResult:
    """Assign prediction slots to ground-truth onsets within a time tolerance.

    Uses the Hungarian algorithm on per-pair absolute time error. Only pairs with
    ``|pred_time - gt_time| <= tolerance_sec`` are valid matches.

    Args:
        pred_times: Predicted onset times in seconds; shape ``(num_queries,)`` or
            ``(batch, num_queries)``.
        gt_times: Ground-truth onset times in seconds; shape ``(n_max_onsets,)`` or
            ``(batch, n_max_onsets)``.
        gt_mask: Mask marking real GT onsets (1) vs padding (0); same shape as
            ``gt_times``.
        tolerance_sec: Maximum allowed absolute time error for a valid match.
        pred_confidence: Optional prediction confidences; accepted for API
            compatibility but not used in v1 matching cost.

    Returns:
        ``MatchResult`` with matched indices and unmatched masks per batch item.
    """
    _ = pred_confidence
    if tolerance_sec < 0:
        raise ValueError("tolerance_sec must be non-negative")

    pred, gt, mask, batch_size, num_queries, n_max = _as_batch(
        pred_times, gt_times, gt_mask
    )
    max_matches = min(num_queries, n_max)

    matched_pred_indices = np.full((batch_size, max_matches), -1, dtype=np.int32)
    matched_gt_indices = np.full((batch_size, max_matches), -1, dtype=np.int32)
    num_matches = np.zeros(batch_size, dtype=np.int32)
    pred_unmatched_mask = np.ones((batch_size, num_queries), dtype=bool)
    gt_unmatched_mask = np.zeros((batch_size, n_max), dtype=bool)

    for batch_idx in range(batch_size):
        (
            matched_pred_indices[batch_idx],
            matched_gt_indices[batch_idx],
            num_matches[batch_idx],
            pred_unmatched_mask[batch_idx],
            gt_unmatched_mask[batch_idx],
        ) = _match_single(
            pred[batch_idx],
            gt[batch_idx],
            mask[batch_idx],
            tolerance_sec,
        )

    return MatchResult(
        matched_pred_indices=matched_pred_indices,
        matched_gt_indices=matched_gt_indices,
        num_matches=num_matches,
        pred_unmatched_mask=pred_unmatched_mask,
        gt_unmatched_mask=gt_unmatched_mask,
    )


def _assign_single_ordered(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Pair query slot ``i`` with the ``i``-th valid GT onset sorted by time."""
    num_queries = pred_times.shape[0]
    n_max = gt_times.shape[0]
    max_matches = min(num_queries, n_max)

    valid_gt = gt_mask.astype(bool)
    gt_slot_indices = np.flatnonzero(valid_gt)
    if gt_slot_indices.size:
        order = np.argsort(gt_times[gt_slot_indices])
        gt_slot_indices = gt_slot_indices[order]

    num_valid_gt = gt_slot_indices.size
    num_pairs = min(num_queries, num_valid_gt)

    matched_pred = np.full(max_matches, -1, dtype=np.int32)
    matched_gt = np.full(max_matches, -1, dtype=np.int32)
    pred_unmatched = np.ones(num_queries, dtype=bool)
    gt_unmatched = np.zeros(n_max, dtype=bool)

    for pair_idx in range(num_pairs):
        matched_pred[pair_idx] = pair_idx
        gt_slot = int(gt_slot_indices[pair_idx])
        matched_gt[pair_idx] = gt_slot
        pred_unmatched[pair_idx] = False

    for gt_idx in range(num_pairs, num_valid_gt):
        gt_unmatched[int(gt_slot_indices[gt_idx])] = True

    return matched_pred, matched_gt, num_pairs, pred_unmatched, gt_unmatched


def assign_onset_pairs_ordered_numpy(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    pred_confidence: np.ndarray | None = None,
) -> MatchResult:
    """Assign query slot ``i`` to the ``i``-th valid GT onset sorted by time.

    Matches the uniform query-time grid in :class:`~stepcovnet.onset_events.models.QuerySpreadTimeNorm`
    and avoids Hungarian reassignment churn during training when predictions cross.

    Args:
        pred_times: Predicted onset times in seconds; shape ``(num_queries,)`` or
            ``(batch, num_queries)``.
        gt_times: Ground-truth onset times in seconds; shape ``(n_max_onsets,)`` or
            ``(batch, n_max_onsets)``.
        gt_mask: Mask marking real GT onsets (1) vs padding (0); same shape as
            ``gt_times``.
        pred_confidence: Optional prediction confidences; accepted for API
            compatibility but not used in assignment.

    Returns:
        ``MatchResult`` with the same fields as :func:`match_onsets_numpy`.
    """
    _ = pred_confidence
    pred, gt, mask, batch_size, num_queries, n_max = _as_batch(
        pred_times, gt_times, gt_mask
    )
    max_matches = min(num_queries, n_max)

    matched_pred_indices = np.full((batch_size, max_matches), -1, dtype=np.int32)
    matched_gt_indices = np.full((batch_size, max_matches), -1, dtype=np.int32)
    num_matches = np.zeros(batch_size, dtype=np.int32)
    pred_unmatched_mask = np.ones((batch_size, num_queries), dtype=bool)
    gt_unmatched_mask = np.zeros((batch_size, n_max), dtype=bool)

    for batch_idx in range(batch_size):
        (
            matched_pred_indices[batch_idx],
            matched_gt_indices[batch_idx],
            num_matches[batch_idx],
            pred_unmatched_mask[batch_idx],
            gt_unmatched_mask[batch_idx],
        ) = _assign_single_ordered(
            pred[batch_idx],
            gt[batch_idx],
            mask[batch_idx],
        )

    return MatchResult(
        matched_pred_indices=matched_pred_indices,
        matched_gt_indices=matched_gt_indices,
        num_matches=num_matches,
        pred_unmatched_mask=pred_unmatched_mask,
        gt_unmatched_mask=gt_unmatched_mask,
    )


def assign_onset_pairs_l1_numpy(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    pred_confidence: np.ndarray | None = None,
) -> MatchResult:
    """Assign prediction slots to GT onsets by minimum total L1 (no tolerance gate).

    Used for training losses so each valid ground-truth onset maps to a unique query
    slot. Inference and metrics still use :func:`match_onsets_numpy` with a time
    tolerance.

    Args:
        pred_times: Predicted onset times in seconds; shape ``(num_queries,)`` or
            ``(batch, num_queries)``.
        gt_times: Ground-truth onset times in seconds; shape ``(n_max_onsets,)`` or
            ``(batch, n_max_onsets)``.
        gt_mask: Mask marking real GT onsets (1) vs padding (0); same shape as
            ``gt_times``.
        pred_confidence: Optional prediction confidences; accepted for API
            compatibility but not used in assignment cost.

    Returns:
        ``MatchResult`` with the same fields as :func:`match_onsets_numpy`.
    """
    _ = pred_confidence
    pred, gt, mask, batch_size, num_queries, n_max = _as_batch(
        pred_times, gt_times, gt_mask
    )
    max_matches = min(num_queries, n_max)

    matched_pred_indices = np.full((batch_size, max_matches), -1, dtype=np.int32)
    matched_gt_indices = np.full((batch_size, max_matches), -1, dtype=np.int32)
    num_matches = np.zeros(batch_size, dtype=np.int32)
    pred_unmatched_mask = np.ones((batch_size, num_queries), dtype=bool)
    gt_unmatched_mask = np.zeros((batch_size, n_max), dtype=bool)

    for batch_idx in range(batch_size):
        (
            matched_pred_indices[batch_idx],
            matched_gt_indices[batch_idx],
            num_matches[batch_idx],
            pred_unmatched_mask[batch_idx],
            gt_unmatched_mask[batch_idx],
        ) = _assign_single_l1(
            pred[batch_idx],
            gt[batch_idx],
            mask[batch_idx],
        )

    return MatchResult(
        matched_pred_indices=matched_pred_indices,
        matched_gt_indices=matched_gt_indices,
        num_matches=num_matches,
        pred_unmatched_mask=pred_unmatched_mask,
        gt_unmatched_mask=gt_unmatched_mask,
    )


def _assign_onset_pairs_ordered_numpy_wrapper(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numpy wrapper for ``tf.numpy_function`` ordered training assignment."""
    result = assign_onset_pairs_ordered_numpy(pred_times, gt_times, gt_mask)
    return (
        result.matched_pred_indices.astype(np.int32),
        result.matched_gt_indices.astype(np.int32),
        result.num_matches.astype(np.int32),
        result.pred_unmatched_mask.astype(np.bool_),
        result.gt_unmatched_mask.astype(np.bool_),
    )


def _assign_onset_pairs_l1_numpy_wrapper(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numpy wrapper for ``tf.numpy_function`` L1 training assignment."""
    result = assign_onset_pairs_l1_numpy(pred_times, gt_times, gt_mask)
    return (
        result.matched_pred_indices.astype(np.int32),
        result.matched_gt_indices.astype(np.int32),
        result.num_matches.astype(np.int32),
        result.pred_unmatched_mask.astype(np.bool_),
        result.gt_unmatched_mask.astype(np.bool_),
    )


def assign_onset_pairs_ordered(
    pred_times: tf.Tensor,
    gt_times: tf.Tensor,
    gt_mask: tf.Tensor,
    pred_confidence: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """TensorFlow ordered assignment for training losses.

    Wraps :func:`assign_onset_pairs_ordered_numpy` with ``tf.numpy_function``.

    Args:
        pred_times: Predicted onset times; shape ``(batch, num_queries)``.
        gt_times: Ground-truth onset times; shape ``(batch, n_max_onsets)``.
        gt_mask: Ground-truth validity mask; same shape as ``gt_times``.
        pred_confidence: Optional prediction confidences; accepted for API
            compatibility but not used in assignment.

    Returns:
        Tuple of ``matched_pred_indices``, ``matched_gt_indices``, ``num_matches``,
        ``pred_unmatched_mask``, and ``gt_unmatched_mask``.
    """
    _ = pred_confidence
    pred_times = tf.convert_to_tensor(pred_times, dtype=tf.float32)
    gt_times = tf.convert_to_tensor(gt_times, dtype=tf.float32)
    gt_mask = tf.convert_to_tensor(gt_mask, dtype=tf.float32)

    pred_shape = tf.shape(pred_times)
    gt_shape = tf.shape(gt_times)
    batch_size = pred_shape[0]
    num_queries = pred_shape[1]
    n_max = gt_shape[1]
    max_matches = tf.minimum(num_queries, n_max)

    outputs = tf.numpy_function(
        _assign_onset_pairs_ordered_numpy_wrapper,
        [pred_times, gt_times, gt_mask],
        [tf.int32, tf.int32, tf.int32, tf.bool, tf.bool],
    )
    (
        matched_pred_indices,
        matched_gt_indices,
        num_matches,
        pred_unmatched,
        gt_unmatched,
    ) = outputs

    matched_pred_indices.set_shape([None, None])
    matched_gt_indices.set_shape([None, None])
    num_matches.set_shape([None])
    pred_unmatched.set_shape([None, None])
    gt_unmatched.set_shape([None, None])

    matched_pred_indices = tf.ensure_shape(
        matched_pred_indices, [batch_size, max_matches]
    )
    matched_gt_indices = tf.ensure_shape(matched_gt_indices, [batch_size, max_matches])
    num_matches = tf.ensure_shape(num_matches, [batch_size])
    pred_unmatched = tf.ensure_shape(pred_unmatched, [batch_size, num_queries])
    gt_unmatched = tf.ensure_shape(gt_unmatched, [batch_size, n_max])

    return (
        matched_pred_indices,
        matched_gt_indices,
        num_matches,
        pred_unmatched,
        gt_unmatched,
    )


def assign_onset_pairs_l1(
    pred_times: tf.Tensor,
    gt_times: tf.Tensor,
    gt_mask: tf.Tensor,
    pred_confidence: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """TensorFlow Hungarian L1 assignment for training losses.

    Wraps :func:`assign_onset_pairs_l1_numpy` with ``tf.numpy_function``.

    Args:
        pred_times: Predicted onset times; shape ``(batch, num_queries)``.
        gt_times: Ground-truth onset times; shape ``(batch, n_max_onsets)``.
        gt_mask: Ground-truth validity mask; same shape as ``gt_times``.
        pred_confidence: Optional prediction confidences; accepted for API
            compatibility but not used in assignment cost.

    Returns:
        Tuple of ``matched_pred_indices``, ``matched_gt_indices``, ``num_matches``,
        ``pred_unmatched_mask``, and ``gt_unmatched_mask``.
    """
    _ = pred_confidence
    pred_times = tf.convert_to_tensor(pred_times, dtype=tf.float32)
    gt_times = tf.convert_to_tensor(gt_times, dtype=tf.float32)
    gt_mask = tf.convert_to_tensor(gt_mask, dtype=tf.float32)

    pred_shape = tf.shape(pred_times)
    gt_shape = tf.shape(gt_times)
    batch_size = pred_shape[0]
    num_queries = pred_shape[1]
    n_max = gt_shape[1]
    max_matches = tf.minimum(num_queries, n_max)

    outputs = tf.numpy_function(
        _assign_onset_pairs_l1_numpy_wrapper,
        [pred_times, gt_times, gt_mask],
        [tf.int32, tf.int32, tf.int32, tf.bool, tf.bool],
    )
    (
        matched_pred_indices,
        matched_gt_indices,
        num_matches,
        pred_unmatched,
        gt_unmatched,
    ) = outputs

    matched_pred_indices.set_shape([None, None])
    matched_gt_indices.set_shape([None, None])
    num_matches.set_shape([None])
    pred_unmatched.set_shape([None, None])
    gt_unmatched.set_shape([None, None])

    matched_pred_indices = tf.ensure_shape(
        matched_pred_indices, [batch_size, max_matches]
    )
    matched_gt_indices = tf.ensure_shape(matched_gt_indices, [batch_size, max_matches])
    num_matches = tf.ensure_shape(num_matches, [batch_size])
    pred_unmatched = tf.ensure_shape(pred_unmatched, [batch_size, num_queries])
    gt_unmatched = tf.ensure_shape(gt_unmatched, [batch_size, n_max])

    return (
        matched_pred_indices,
        matched_gt_indices,
        num_matches,
        pred_unmatched,
        gt_unmatched,
    )


def _match_onsets_numpy_wrapper(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numpy wrapper for ``tf.numpy_function`` (float tolerance as length-1 array)."""
    result = match_onsets_numpy(
        pred_times,
        gt_times,
        gt_mask,
        tolerance_sec=float(tolerance_sec.reshape(-1)[0]),
    )
    return (
        result.matched_pred_indices.astype(np.int32),
        result.matched_gt_indices.astype(np.int32),
        result.num_matches.astype(np.int32),
        result.pred_unmatched_mask.astype(np.bool_),
        result.gt_unmatched_mask.astype(np.bool_),
    )


def match_onsets(
    pred_times: tf.Tensor,
    gt_times: tf.Tensor,
    gt_mask: tf.Tensor,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    pred_confidence: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """TensorFlow Hungarian matching between prediction slots and GT onsets.

    Wraps :func:`match_onsets_numpy` with ``tf.numpy_function`` for v1 training.

    Args:
        pred_times: Predicted onset times; shape ``(batch, num_queries)``.
        gt_times: Ground-truth onset times; shape ``(batch, n_max_onsets)``.
        gt_mask: Ground-truth validity mask; same shape as ``gt_times``.
        tolerance_sec: Maximum allowed absolute time error for a valid match.
        pred_confidence: Optional prediction confidences; accepted for API
            compatibility but not used in v1 matching cost.

    Returns:
        Tuple of ``matched_pred_indices``, ``matched_gt_indices``, ``num_matches``,
        ``pred_unmatched_mask``, and ``gt_unmatched_mask`` with the same semantics
        as :class:`MatchResult`.
    """
    _ = pred_confidence
    pred_times = tf.convert_to_tensor(pred_times, dtype=tf.float32)
    gt_times = tf.convert_to_tensor(gt_times, dtype=tf.float32)
    gt_mask = tf.convert_to_tensor(gt_mask, dtype=tf.float32)

    pred_shape = tf.shape(pred_times)
    gt_shape = tf.shape(gt_times)
    batch_size = pred_shape[0]
    num_queries = pred_shape[1]
    n_max = gt_shape[1]
    max_matches = tf.minimum(num_queries, n_max)

    outputs = tf.numpy_function(
        _match_onsets_numpy_wrapper,
        [
            pred_times,
            gt_times,
            gt_mask,
            np.array([tolerance_sec], dtype=np.float64),
        ],
        [
            tf.int32,
            tf.int32,
            tf.int32,
            tf.bool,
            tf.bool,
        ],
    )
    (
        matched_pred_indices,
        matched_gt_indices,
        num_matches,
        pred_unmatched,
        gt_unmatched,
    ) = outputs

    matched_pred_indices.set_shape([None, None])
    matched_gt_indices.set_shape([None, None])
    num_matches.set_shape([None])
    pred_unmatched.set_shape([None, None])
    gt_unmatched.set_shape([None, None])

    matched_pred_indices = tf.ensure_shape(
        matched_pred_indices, [batch_size, max_matches]
    )
    matched_gt_indices = tf.ensure_shape(matched_gt_indices, [batch_size, max_matches])
    num_matches = tf.ensure_shape(num_matches, [batch_size])
    pred_unmatched = tf.ensure_shape(pred_unmatched, [batch_size, num_queries])
    gt_unmatched = tf.ensure_shape(gt_unmatched, [batch_size, n_max])

    return (
        matched_pred_indices,
        matched_gt_indices,
        num_matches,
        pred_unmatched,
        gt_unmatched,
    )
