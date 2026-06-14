"""Combined classification and time regression loss for event-based onset detection."""

import tensorflow as tf

from stepcovnet.onset_events import matching

_BEYOND_TOLERANCE_TIME_WEIGHT = 5.0


def _binary_crossentropy_to_one(confidence: tf.Tensor) -> tf.Tensor:
    """Per-slot BCE toward 1 without reducing across query slots."""
    return -tf.math.log(confidence)


def _binary_crossentropy_to_zero(confidence: tf.Tensor) -> tf.Tensor:
    """Per-slot BCE toward 0 without reducing across query slots."""
    return -tf.math.log(1.0 - confidence)


def _gather_matched(
    values: tf.Tensor,
    pred_indices: tf.Tensor,
    gt_indices: tf.Tensor,
    num_matches: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Gather matched pred and GT values; return them with a float validity mask."""
    batch_size = tf.shape(values)[0]
    max_matches = tf.shape(pred_indices)[1]
    batch_indices = tf.broadcast_to(
        tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis],
        tf.stack([batch_size, max_matches]),
    )
    match_positions = tf.broadcast_to(
        tf.range(max_matches, dtype=tf.int32)[tf.newaxis, :],
        tf.stack([batch_size, max_matches]),
    )
    valid_match = tf.logical_and(
        match_positions < num_matches[:, tf.newaxis],
        pred_indices >= 0,
    )
    valid_f = tf.cast(valid_match, tf.float32)
    safe_pred = tf.where(valid_match, pred_indices, tf.zeros_like(pred_indices))
    safe_gt = tf.where(valid_match, gt_indices, tf.zeros_like(gt_indices))
    pred_vals = tf.gather_nd(values, tf.stack([batch_indices, safe_pred], axis=-1))
    return pred_vals, safe_gt, valid_f


def _l1_training_losses(
    pred_times: tf.Tensor,
    gt_times: tf.Tensor,
    gt_mask: tf.Tensor,
    pred_confidence: tf.Tensor,
    tolerance_sec: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """L1 time loss and confidence BCE losses from Hungarian L1 assignment.

    Each valid ground-truth onset is paired with the query slot that minimizes total
    L1 cost (same pairing as :func:`~stepcovnet.onset_events.matching.assign_onset_pairs_l1`).
    Assigned pairs receive mean L1 plus L2 time loss. Per matched pair, confidence is
    pushed toward 1 when ``l1`` is within ``tolerance_sec`` and toward 0 otherwise;
    unmatched query slots are pushed toward 0.

    Returns:
        Tuple of ``(time_loss, matched_conf_loss, unmatched_conf_loss)``.
    """
    (
        matched_pred_indices,
        matched_gt_indices,
        num_matches,
        pred_unmatched,
        _gt_unmatched,
    ) = matching.assign_onset_pairs_l1(pred_times, gt_times, gt_mask)
    matched_pred_indices = tf.stop_gradient(matched_pred_indices)
    matched_gt_indices = tf.stop_gradient(matched_gt_indices)
    num_matches = tf.stop_gradient(num_matches)
    pred_unmatched = tf.stop_gradient(pred_unmatched)

    batch_size = tf.shape(pred_times)[0]
    max_matches = tf.shape(matched_pred_indices)[1]
    batch_indices = tf.broadcast_to(
        tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis],
        tf.stack([batch_size, max_matches]),
    )
    match_positions = tf.broadcast_to(
        tf.range(max_matches, dtype=tf.int32)[tf.newaxis, :],
        tf.stack([batch_size, max_matches]),
    )
    valid_match = tf.logical_and(
        match_positions < num_matches[:, tf.newaxis],
        matched_pred_indices >= 0,
    )
    valid_f = tf.cast(valid_match, tf.float32)
    safe_pred = tf.where(
        valid_match, matched_pred_indices, tf.zeros_like(matched_pred_indices)
    )
    safe_gt = tf.where(
        valid_match, matched_gt_indices, tf.zeros_like(matched_gt_indices)
    )

    pred_matched_times = tf.gather_nd(
        pred_times, tf.stack([batch_indices, safe_pred], axis=-1)
    )
    gt_matched_times = tf.gather_nd(
        gt_times, tf.stack([batch_indices, safe_gt], axis=-1)
    )
    l1 = tf.abs(pred_matched_times - gt_matched_times)
    denom = tf.maximum(tf.reduce_sum(valid_f), 1.0)
    mean_l1 = tf.reduce_sum(l1 * valid_f) / denom
    mean_l2 = tf.reduce_sum(l1 * l1 * valid_f) / denom
    beyond_tol = tf.nn.relu(l1 - tolerance_sec)
    mean_beyond = tf.reduce_sum(beyond_tol * valid_f) / denom
    time_loss = mean_l1 + mean_l2 + _BEYOND_TOLERANCE_TIME_WEIGHT * mean_beyond

    num_queries = tf.shape(pred_times)[1]
    within_tol_match = tf.logical_and(valid_match, l1 <= tolerance_sec)
    outside_tol_match = tf.logical_and(valid_match, l1 > tolerance_sec)
    tol_batch = tf.boolean_mask(batch_indices, within_tol_match)
    tol_pred = tf.boolean_mask(safe_pred, within_tol_match)
    tol_count = tf.shape(tol_batch)[0]
    tol_positive = tf.scatter_nd(
        tf.stack([tol_batch, tol_pred], axis=1),
        tf.ones(tol_count, dtype=tf.float32),
        [batch_size, num_queries],
    )
    out_batch = tf.boolean_mask(batch_indices, outside_tol_match)
    out_pred = tf.boolean_mask(safe_pred, outside_tol_match)
    out_count = tf.shape(out_batch)[0]
    tol_negative = tf.scatter_nd(
        tf.stack([out_batch, out_pred], axis=1),
        tf.ones(out_count, dtype=tf.float32),
        [batch_size, num_queries],
    )
    matched_bce = _binary_crossentropy_to_one(pred_confidence)
    unmatched_bce = _binary_crossentropy_to_zero(pred_confidence)
    tol_f = tol_positive
    tol_conf = tf.reduce_sum(matched_bce * tol_f) / tf.maximum(
        tf.reduce_sum(tol_f), 1.0
    )
    outside_f = tol_negative
    outside_conf = tf.reduce_sum(unmatched_bce * outside_f) / tf.maximum(
        tf.reduce_sum(outside_f), 1.0
    )
    matched_conf_loss = tol_conf
    unmatched_f = tf.cast(pred_unmatched, tf.float32)
    extra_unmatched = tf.reduce_sum(unmatched_bce * unmatched_f) / tf.maximum(
        tf.reduce_sum(unmatched_f), 1.0
    )
    unmatched_conf_loss = outside_conf + extra_unmatched
    return time_loss, matched_conf_loss, unmatched_conf_loss


def compute_onset_event_loss(
    pred_times: tf.Tensor,
    pred_confidence: tf.Tensor,
    gt_times: tf.Tensor,
    gt_mask: tf.Tensor,
    duration: tf.Tensor,
    tolerance_sec: float = matching.DEFAULT_TOLERANCE_SEC,
    lambda_cls: float = 1.0,
    lambda_time: float = 5.0,
    return_components: bool = False,
) -> tf.Tensor | tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Compute onset event loss using Hungarian L1 query-to-GT assignment.

    Each valid ground-truth onset is paired with the query slot that minimizes total
    L1 cost, aligning training with eval matching semantics (without the tolerance
    gate on assignment). Confidence follows matched-pair time error: toward 1 within
    ``tolerance_sec``, toward 0 otherwise.

    Args:
        pred_times: Predicted onset times in seconds; shape ``(batch, num_queries)``.
        pred_confidence: Predicted slot confidences in ``[0, 1]``; same shape as
            ``pred_times``.
        gt_times: Ground-truth onset times in seconds; shape ``(batch, n_max_onsets)``.
        gt_mask: Ground-truth validity mask (1 = real onset); same shape as ``gt_times``.
        duration: Song duration in seconds per batch item; shape ``(batch,)`` or scalar.
            Accepted for training-batch compatibility; not used in the loss formula.
        tolerance_sec: Pairs within this slack receive confidence training toward 1.
        lambda_cls: Weight on matched and unmatched confidence BCE terms.
        lambda_time: Weight on L1 time error for matched pairs.
        return_components: When ``True``, also return unweighted loss tensors in a
            dict.

    Returns:
        Scalar total loss, or ``(total_loss, components)`` when ``return_components``
        is ``True``.
    """
    _ = duration
    pred_times = tf.cast(tf.convert_to_tensor(pred_times), tf.float32)
    pred_confidence = tf.clip_by_value(
        tf.cast(tf.convert_to_tensor(pred_confidence), tf.float32),
        1e-4,
        1.0 - 1e-4,
    )
    gt_times = tf.convert_to_tensor(gt_times, dtype=tf.float32)
    gt_mask = tf.convert_to_tensor(gt_mask, dtype=tf.float32)

    time_loss, matched_conf_loss, unmatched_conf_loss = _l1_training_losses(
        pred_times,
        gt_times,
        gt_mask,
        pred_confidence,
        tolerance_sec,
    )

    total_loss = (
        lambda_cls * (matched_conf_loss + 0.25 * unmatched_conf_loss)
        + lambda_time * time_loss
    )

    if not return_components:
        return total_loss

    components = {
        "matched_conf_loss": matched_conf_loss,
        "unmatched_conf_loss": unmatched_conf_loss,
        "time_loss": time_loss,
        "total_loss": total_loss,
    }
    return total_loss, components
