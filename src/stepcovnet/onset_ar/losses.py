"""Losses for AR onset training (token CE + pointer/residual time)."""

from __future__ import annotations

import tensorflow as tf


def predicted_times_from_outputs(
    pointer_logits: tf.Tensor,
    residual_sec: tf.Tensor,
    *,
    patch_frames: int,
    hop_sec: float,
) -> tf.Tensor:
    """Expected onset times from pointer logits and residual head."""
    patch_frames_f = tf.cast(patch_frames, tf.float32)
    hop_sec_f = tf.cast(hop_sec, tf.float32)
    patch_duration = patch_frames_f * hop_sec_f
    n_patches = tf.shape(pointer_logits)[-1]
    patch_indices = tf.cast(tf.range(n_patches), tf.float32)
    probs = tf.nn.softmax(pointer_logits, axis=-1)
    expected_patch = tf.reduce_sum(probs * patch_indices, axis=-1)
    return expected_patch * patch_duration + residual_sec


def compute_ar_onset_loss(
    outputs: dict[str, tf.Tensor],
    batch: dict[str, tf.Tensor],
    *,
    patch_frames: int,
    hop_sec: float,
    lambda_time: float,
    length_normalize_ce: bool,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Combined teacher-forcing loss for ``gate-tide-overfit``."""
    token_logits = outputs["token_logits"]
    pointer_logits = outputs["pointer_logits"]
    residual_sec = outputs["residual_sec"]

    decoder_target_ids = tf.cast(batch["decoder_target_ids"], tf.int32)
    decoder_mask = tf.cast(batch["decoder_mask"], tf.float32)
    onset_step_mask = tf.cast(batch["onset_step_mask"], tf.float32)
    target_patch_indices = tf.cast(batch["target_patch_indices"], tf.int32)
    target_times = tf.cast(batch["target_times"], tf.float32)

    token_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=decoder_target_ids,
        logits=token_logits,
    )
    token_losses = token_losses * decoder_mask
    token_count = tf.reduce_sum(decoder_mask) + 1e-9
    if length_normalize_ce:
        token_loss = tf.reduce_sum(token_losses) / token_count
    else:
        token_loss = tf.reduce_sum(token_losses)

    pointer_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_patch_indices,
        logits=pointer_logits,
    )
    pointer_losses = pointer_losses * onset_step_mask
    pointer_count = tf.reduce_sum(onset_step_mask) + 1e-9
    pointer_loss = tf.reduce_sum(pointer_losses) / pointer_count

    pred_times = predicted_times_from_outputs(
        pointer_logits,
        residual_sec,
        patch_frames=patch_frames,
        hop_sec=hop_sec,
    )
    time_errors = tf.abs(pred_times - target_times) * onset_step_mask
    time_loss = tf.reduce_sum(time_errors) / pointer_count

    total_loss = (
        token_loss + pointer_loss + tf.cast(lambda_time, tf.float32) * time_loss
    )
    return total_loss, {
        "token_loss": token_loss,
        "pointer_loss": pointer_loss,
        "time_loss": time_loss,
    }


def masked_token_accuracy(
    token_logits: tf.Tensor,
    decoder_target_ids: tf.Tensor,
    decoder_mask: tf.Tensor,
) -> tf.Tensor:
    """Token accuracy over non-padded decoder positions."""
    predictions = tf.argmax(token_logits, axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(predictions, decoder_target_ids), tf.float32)
    correct = correct * decoder_mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(decoder_mask) + 1e-9)
