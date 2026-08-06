"""Losses for AR onset training (token CE + pointer/residual time)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import pointer_mask, targets


def build_token_class_weights_numpy(
    decoder_target_ids: np.ndarray,
    decoder_mask: np.ndarray,
    *,
    vocab_size: int,
    scheme: str,
    eos_token_weight_scale: float = 1.0,
) -> np.ndarray | None:
    """Return per-vocab-id CE weights for the active decoder targets."""
    if scheme == "none":
        return None
    mask = np.asarray(decoder_mask, dtype=np.float64).reshape(-1) > 0.5
    target_ids = np.asarray(decoder_target_ids, dtype=np.int64).reshape(-1)[mask]
    counts = np.bincount(target_ids, minlength=vocab_size).astype(np.float64)
    freq = counts / max(float(counts.sum()), 1.0)
    if scheme == "inverse_freq":
        weights = 1.0 / np.maximum(freq, 1e-6)
    elif scheme == "inverse_sqrt_freq":
        weights = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    else:
        raise ValueError(f"unsupported token_class_weight scheme: {scheme!r}")
    weights = weights / max(float(weights.mean()), 1e-6)
    if eos_token_weight_scale != 1.0:
        weights[targets.EOS_ID] *= float(eos_token_weight_scale)
    return weights.astype(np.float32)


def predicted_times_from_outputs(
    pointer_logits: tf.Tensor,
    residual_sec: tf.Tensor,
    *,
    patch_frames: int,
    hop_sec: float,
    use_soft_expected: bool = False,
) -> tf.Tensor:
    """Convert pointer logits and residual head outputs to onset times in seconds."""
    pointer_logits = tf.cast(pointer_logits, tf.float32)
    residual_sec = tf.cast(residual_sec, tf.float32)
    patch_frames_f = tf.cast(patch_frames, tf.float32)
    hop_sec_f = tf.cast(hop_sec, tf.float32)
    patch_duration = patch_frames_f * hop_sec_f
    if use_soft_expected:
        n_patches = tf.shape(pointer_logits)[-1]
        patch_indices = tf.cast(tf.range(n_patches), tf.float32)
        probs = tf.nn.softmax(pointer_logits, axis=-1)
        expected_patch = tf.reduce_sum(probs * patch_indices, axis=-1)
        patch_idx = expected_patch
    else:
        patch_idx = tf.cast(
            tf.argmax(pointer_logits, axis=-1),
            tf.float32,
        )
    return patch_idx * patch_duration + residual_sec


def predicted_time_at_decoder_position(
    pointer_logits: tf.Tensor,
    residual_sec: tf.Tensor,
    *,
    patch_frames: int,
    hop_sec: float,
    use_soft_expected: bool = False,
) -> tf.Tensor:
    """Scalar onset time per batch item from one decoder position."""
    pointer_logits = tf.expand_dims(pointer_logits, axis=1)
    residual_sec = tf.expand_dims(residual_sec, axis=1)
    times = predicted_times_from_outputs(
        pointer_logits,
        residual_sec,
        patch_frames=patch_frames,
        hop_sec=hop_sec,
        use_soft_expected=use_soft_expected,
    )
    return times[:, 0]


def incremental_predicted_times_tf(
    decoder: tf.keras.Model,
    encoder_memory: tf.Tensor,
    patch_mask: tf.Tensor,
    decoder_input_ids: tf.Tensor,
    decoder_mask: tf.Tensor,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    use_soft_pointer_time: bool = False,
    max_unroll_steps: int = 0,
) -> tf.Tensor:
    """Prefix decoder unrolls; predicted time written at each visited position."""
    batch_size = tf.shape(decoder_input_ids)[0]
    times = tf.zeros((batch_size, max_decoder_len), dtype=tf.float32)
    seq_len = tf.cast(tf.reduce_sum(decoder_mask[0]), tf.int32)
    if max_unroll_steps > 0:
        seq_len = tf.minimum(seq_len, tf.cast(max_unroll_steps, tf.int32))

    def cond(cur_len: tf.Tensor, _times: tf.Tensor) -> tf.Tensor:
        return cur_len <= seq_len

    def body(cur_len: tf.Tensor, step_times: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        cur_len_f = tf.cast(cur_len, tf.float32)
        positions = tf.cast(tf.range(max_decoder_len), tf.float32)[tf.newaxis, :]
        prefix_mask = tf.cast(positions < cur_len_f, tf.float32) * decoder_mask
        outputs = decoder(
            {
                "encoder_memory": encoder_memory,
                "patch_mask": patch_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_mask": prefix_mask,
            },
            training=True,
        )
        pos = cur_len - 1
        step_time = predicted_time_at_decoder_position(
            outputs["pointer_logits"][:, pos, :],
            outputs["residual_sec"][:, pos],
            patch_frames=patch_frames,
            hop_sec=hop_sec,
            use_soft_expected=use_soft_pointer_time,
        )
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        scatter_indices = tf.stack(
            [batch_indices, tf.fill((batch_size,), pos)],
            axis=1,
        )
        step_times = tf.tensor_scatter_nd_update(
            step_times,
            scatter_indices,
            step_time,
        )
        return cur_len + 1, step_times

    _, times = tf.while_loop(
        cond,
        body,
        (tf.constant(1, dtype=tf.int32), times),
        maximum_iterations=max_decoder_len,
    )
    return times


def sampled_incremental_consistency_loss_tf(
    decoder: tf.keras.Model,
    encoder_memory: tf.Tensor,
    patch_mask: tf.Tensor,
    decoder_input_ids: tf.Tensor,
    decoder_mask: tf.Tensor,
    parallel_times: tf.Tensor,
    onset_step_mask: tf.Tensor,
    *,
    max_decoder_len: int,
    patch_frames: int,
    hop_sec: float,
    use_soft_pointer_time: bool = False,
    n_samples: int,
    pointer_key_input: tf.Tensor | None = None,
) -> tf.Tensor:
    """L_inc at one random onset position per step (GPU-safe on tide).

    ``n_samples`` is reserved for future gradient accumulation; only one prefix
    decode is differentiated per train step so cross-attn is not stacked in-graph.
    """
    del n_samples
    parallel_times = tf.stop_gradient(parallel_times)
    encoder_memory = tf.stop_gradient(encoder_memory)
    key_input = (
        encoder_memory
        if pointer_key_input is None
        else tf.stop_gradient(pointer_key_input)
    )
    onset_positions = tf.reshape(
        tf.where(onset_step_mask[0] > 0.5)[:, 0],
        (-1,),
    )
    n_onsets = tf.shape(onset_positions)[0]

    def _zero_loss() -> tf.Tensor:
        return tf.constant(0.0, dtype=tf.float32)

    def _single_sample_loss() -> tf.Tensor:
        pos = tf.random.shuffle(onset_positions)[0]
        cur_len_f = tf.cast(pos + 1, tf.float32)
        positions = tf.cast(tf.range(max_decoder_len), tf.float32)[tf.newaxis, :]
        prefix_mask = tf.cast(positions < cur_len_f, tf.float32) * decoder_mask
        decoder_inputs = {
            "encoder_memory": encoder_memory,
            "patch_mask": patch_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_mask": prefix_mask,
        }
        decoder_input_map = getattr(decoder, "input", None)
        if (
            isinstance(decoder_input_map, dict)
            and "pointer_key_input" in decoder_input_map
        ):
            decoder_inputs["pointer_key_input"] = key_input
        outputs = decoder(
            decoder_inputs,
            training=True,
        )
        inc_time = predicted_time_at_decoder_position(
            outputs["pointer_logits"][:, pos, :],
            outputs["residual_sec"][:, pos],
            patch_frames=patch_frames,
            hop_sec=hop_sec,
            use_soft_expected=use_soft_pointer_time,
        )
        par_time = parallel_times[0, pos]
        return tf.abs(par_time - inc_time[0])

    return tf.cond(n_onsets > 0, _single_sample_loss, _zero_loss)


def incremental_consistency_loss(
    parallel_times: tf.Tensor,
    incremental_times: tf.Tensor,
    onset_step_mask: tf.Tensor,
) -> tf.Tensor:
    """Mean absolute gap between parallel and prefix-incremental predicted times."""
    parallel_times = tf.stop_gradient(parallel_times)
    mask = tf.cast(onset_step_mask, tf.float32)
    diff = tf.abs(parallel_times - incremental_times) * mask
    count = tf.reduce_sum(mask) + 1e-9
    return tf.reduce_sum(diff) / count


def compute_ar_onset_loss(
    outputs: dict[str, tf.Tensor],
    batch: dict[str, tf.Tensor],
    *,
    patch_frames: int,
    hop_sec: float,
    lambda_time: float,
    lambda_residual: float,
    pointer_loss_weight: float,
    length_normalize_ce: bool,
    token_class_weights: tf.Tensor | None = None,
    use_soft_pointer_time: bool = False,
    monotonic_pointer: bool = False,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Combined teacher-forcing loss for ``gate-tide-overfit``."""
    token_logits = tf.cast(outputs["token_logits"], tf.float32)
    pointer_logits = tf.cast(outputs["pointer_logits"], tf.float32)
    residual_sec = tf.cast(outputs["residual_sec"], tf.float32)

    decoder_target_ids = tf.cast(batch["decoder_target_ids"], tf.int32)
    decoder_mask = tf.cast(batch["decoder_mask"], tf.float32)
    onset_step_mask = tf.cast(batch["onset_step_mask"], tf.float32)
    target_patch_indices = tf.cast(batch["target_patch_indices"], tf.int32)
    target_times = tf.cast(batch["target_times"], tf.float32)
    target_residual_sec = tf.cast(batch["target_residual_sec"], tf.float32)

    if monotonic_pointer:
        prev_patches = pointer_mask.teacher_forced_prev_patch_indices(
            target_patch_indices,
        )
        pointer_logits = pointer_mask.apply_monotonic_pointer_mask_tf(
            pointer_logits,
            prev_patches,
        )

    token_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=decoder_target_ids,
        logits=token_logits,
    )
    if token_class_weights is not None:
        per_token_weight = tf.gather(token_class_weights, decoder_target_ids)
        token_losses = token_losses * per_token_weight
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
        use_soft_expected=use_soft_pointer_time,
    )
    time_errors = tf.abs(pred_times - target_times) * onset_step_mask
    time_loss = tf.reduce_sum(time_errors) / pointer_count

    residual_sq = tf.square(residual_sec - target_residual_sec) * onset_step_mask
    residual_loss = tf.reduce_sum(residual_sq) / pointer_count

    pointer_term = tf.cast(pointer_loss_weight, tf.float32) * pointer_loss
    total_loss = (
        token_loss
        + pointer_term
        + tf.cast(lambda_time, tf.float32) * time_loss
        + tf.cast(lambda_residual, tf.float32) * residual_loss
    )
    return total_loss, {
        "token_loss": token_loss,
        "pointer_loss": pointer_loss,
        "time_loss": time_loss,
        "residual_loss": residual_loss,
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
