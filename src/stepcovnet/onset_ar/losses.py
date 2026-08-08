"""Losses for AR onset training (token CE + pointer/gap/residual time)."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import pointer_mask, targets


def resolved_patches_from_gap_logits(
    gap_logits: tf.Tensor,
    prev_patch_indices: tf.Tensor,
    *,
    delta_lookup: tf.Tensor,
    max_patch: tf.Tensor | int,
) -> tf.Tensor:
    """Resolve absolute patches as ``prev + decode(argmax Δ)``, clamped.

    Args:
        gap_logits: Gap-vocab logits ``[B, T, V]``.
        prev_patch_indices: Teacher-forced previous patch ``[B, T]`` (0 at step 0).
        delta_lookup: ``decode_delta(id)`` table ``[V]``.
        max_patch: Inclusive clamp (typically ``T' - 1``).

    Returns:
        Absolute patch indices ``[B, T]``.
    """
    gap_ids = tf.argmax(gap_logits, axis=-1, output_type=tf.int32)
    deltas = tf.gather(tf.cast(delta_lookup, tf.int32), gap_ids)
    prev = tf.cast(prev_patch_indices, tf.int32)
    patches = prev + deltas
    max_patch_i = tf.cast(max_patch, tf.int32)
    return tf.clip_by_value(patches, 0, max_patch_i)


def predicted_times_from_patches(
    patch_indices: tf.Tensor,
    residual_sec: tf.Tensor,
    *,
    patch_frames: int,
    hop_sec: float,
) -> tf.Tensor:
    """Convert absolute patch indices and residual seconds to onset times."""
    patch_indices = tf.cast(patch_indices, tf.float32)
    residual_sec = tf.cast(residual_sec, tf.float32)
    patch_duration = tf.cast(patch_frames, tf.float32) * tf.cast(hop_sec, tf.float32)
    return patch_indices * patch_duration + residual_sec


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


def _soft_expected_patch_indices(pointer_logits: tf.Tensor) -> tf.Tensor:
    """Softmax-weighted expected patch index per decoder step."""
    n_patches = tf.shape(pointer_logits)[-1]
    patch_indices = tf.cast(tf.range(n_patches), tf.float32)
    probs = tf.nn.softmax(pointer_logits, axis=-1)
    return tf.reduce_sum(probs * patch_indices, axis=-1)


def predicted_times_from_outputs(
    pointer_logits: tf.Tensor,
    residual_sec: tf.Tensor,
    *,
    patch_frames: int,
    hop_sec: float,
    use_soft_expected: bool = False,
    use_ste: bool = False,
) -> tf.Tensor:
    """Convert pointer logits and residual head outputs to onset times in seconds.

    Args:
        pointer_logits: Pointer logits ``[B, T, P]``.
        residual_sec: Within-patch residual seconds ``[B, T]``.
        patch_frames: Frames per patch.
        hop_sec: Frame hop in seconds.
        use_soft_expected: If True (and not ``use_ste``), use soft expected patch.
        use_ste: If True, hard argmax forward with soft expected backward (STE).
    """
    pointer_logits = tf.cast(pointer_logits, tf.float32)
    residual_sec = tf.cast(residual_sec, tf.float32)
    patch_frames_f = tf.cast(patch_frames, tf.float32)
    hop_sec_f = tf.cast(hop_sec, tf.float32)
    patch_duration = patch_frames_f * hop_sec_f
    hard_patch = tf.cast(tf.argmax(pointer_logits, axis=-1), tf.float32)
    if use_ste:
        soft_patch = _soft_expected_patch_indices(pointer_logits)
        # Forward = hard; backward = soft (straight-through estimator).
        patch_idx = soft_patch + tf.stop_gradient(hard_patch - soft_patch)
    elif use_soft_expected:
        patch_idx = _soft_expected_patch_indices(pointer_logits)
    else:
        patch_idx = hard_patch
    return patch_idx * patch_duration + residual_sec


def apply_local_pointer_ce_mask(
    pointer_logits: tf.Tensor,
    target_patch_indices: tf.Tensor,
    *,
    radius: int,
) -> tf.Tensor:
    """Mask pointer logits outside ``[target - radius, target + radius]``.

    Args:
        pointer_logits: Pointer logits ``[B, T, P]``.
        target_patch_indices: Target patch ids ``[B, T]``.
        radius: Inclusive half-width in patches. ``0`` or negative leaves logits
            unchanged.

    Returns:
        Logits with out-of-window positions set to a large negative value.
    """
    if radius <= 0:
        return pointer_logits
    pointer_logits = tf.cast(pointer_logits, tf.float32)
    target_patch_indices = tf.cast(target_patch_indices, tf.int32)
    n_patches = tf.shape(pointer_logits)[-1]
    patch_ids = tf.range(n_patches, dtype=tf.int32)
    patch_ids = tf.reshape(patch_ids, (1, 1, n_patches))
    target = tf.expand_dims(target_patch_indices, axis=-1)
    lo = target - int(radius)
    hi = target + int(radius)
    invalid = tf.logical_or(patch_ids < lo, patch_ids > hi)
    return pointer_logits + tf.cast(invalid, pointer_logits.dtype) * (-1e9)


def predicted_time_at_decoder_position(
    pointer_logits: tf.Tensor,
    residual_sec: tf.Tensor,
    *,
    patch_frames: int,
    hop_sec: float,
    use_soft_expected: bool = False,
    use_ste: bool = False,
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
        use_ste=use_ste,
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
    pointer_key_input: tf.Tensor | None = None,
) -> tf.Tensor:
    """Prefix decoder unrolls; predicted time written at each visited position."""
    batch_size = tf.shape(decoder_input_ids)[0]
    times = tf.zeros((batch_size, max_decoder_len), dtype=tf.float32)
    seq_len = tf.cast(tf.reduce_sum(decoder_mask[0]), tf.int32)
    if max_unroll_steps > 0:
        seq_len = tf.minimum(seq_len, tf.cast(max_unroll_steps, tf.int32))
    key_input = encoder_memory if pointer_key_input is None else pointer_key_input

    def cond(cur_len: tf.Tensor, _times: tf.Tensor) -> tf.Tensor:
        return cur_len <= seq_len

    def body(cur_len: tf.Tensor, step_times: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        cur_len_f = tf.cast(cur_len, tf.float32)
        positions = tf.cast(tf.range(max_decoder_len), tf.float32)[tf.newaxis, :]
        prefix_mask = tf.cast(positions < cur_len_f, tf.float32) * decoder_mask
        decoder_feed: dict[str, tf.Tensor] = {
            "encoder_memory": encoder_memory,
            "patch_mask": patch_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_mask": prefix_mask,
        }
        dec_inputs = decoder.input
        if isinstance(dec_inputs, dict) and "pointer_key_input" in dec_inputs:
            decoder_feed["pointer_key_input"] = key_input
        outputs = decoder(
            decoder_feed,
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
    use_ste_pointer_time: bool = False,
    time_loss_correct_patch_only: bool = False,
    pointer_local_ce_radius: int = 0,
    pointer_local_ce_anchor: str = "target",
    pointer_soft_distance_alpha: float | tf.Tensor = 0.0,
    monotonic_pointer: bool = False,
    gap_alignment: bool = False,
    gap_loss_weight: float = 1.0,
    gap_delta_lookup: tf.Tensor | None = None,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    """Combined teacher-forcing loss for ``gate-tide-overfit``.

    Args:
        outputs: Model heads (``token_logits``, ``pointer_logits`` and/or
            ``gap_logits``, ``residual_sec``).
        batch: Teacher-forced batch tensors.
        patch_frames: Frames per patch.
        hop_sec: Frame hop in seconds.
        lambda_time: Weight on absolute time error.
        lambda_residual: Weight on residual MSE.
        pointer_loss_weight: Weight on absolute pointer CE (A/B or legacy).
        length_normalize_ce: Mean token CE over valid positions when True.
        token_class_weights: Optional per-vocab CE weights.
        use_soft_pointer_time: Soft expected patch for ``λ_time`` (ignored if STE).
        use_ste_pointer_time: Hard forward / soft backward for ``λ_time``.
        time_loss_correct_patch_only: Apply ``λ_time`` only where hard argmax
            matches the target patch (avoids residual fighting wrong patches).
        pointer_local_ce_radius: If ``> 0``, restrict pointer CE to a window
            (see ``pointer_local_ce_anchor``). Ignored when ``gap_alignment``.
        pointer_local_ce_anchor: ``\"target\"`` → ``[target±r]`` CE-only;
            ``\"prev\"`` → ``[prev, prev+r]`` (also applied to time-head logits);
            teacher gaps ``> r`` are dropped from pointer CE (not poisoned).
        pointer_soft_distance_alpha: Soft ahead / Δ penalty (float or live
            ``tf.Variable`` for anneal). Absolute pointer: ``α·max(0, p−prev)``.
            Gap head: ``α·decode_delta(id)`` (skipped when ``prev == 0``).
            Hard-R still ignored on the gap path.
        monotonic_pointer: Apply teacher-forced monotonic mask on absolute
            pointer logits before losses.
        gap_alignment: Use relative gap CE as primary alignment loss.
        gap_loss_weight: Weight on gap CE when ``gap_alignment``.
        gap_delta_lookup: ``decode_delta(id)`` table for resolving patches.
    """
    token_logits = tf.cast(outputs["token_logits"], tf.float32)
    residual_sec = tf.cast(outputs["residual_sec"], tf.float32)
    has_pointer = "pointer_logits" in outputs
    has_gap = "gap_logits" in outputs
    pointer_logits = (
        tf.cast(outputs["pointer_logits"], tf.float32) if has_pointer else None
    )
    gap_logits = tf.cast(outputs["gap_logits"], tf.float32) if has_gap else None

    decoder_target_ids = tf.cast(batch["decoder_target_ids"], tf.int32)
    decoder_mask = tf.cast(batch["decoder_mask"], tf.float32)
    onset_step_mask = tf.cast(batch["onset_step_mask"], tf.float32)
    target_patch_indices = tf.cast(batch["target_patch_indices"], tf.int32)
    target_times = tf.cast(batch["target_times"], tf.float32)
    target_residual_sec = tf.cast(batch["target_residual_sec"], tf.float32)

    prev_patches = pointer_mask.teacher_forced_prev_patch_indices(
        target_patch_indices,
    )
    # Cast (not float()) so a live anneal Variable stays in the train graph.
    soft_alpha = tf.cast(pointer_soft_distance_alpha, tf.float32)
    # Hard-R stays diagnostic-only and is not applied on the gap path.
    if pointer_logits is not None and not gap_alignment:
        if monotonic_pointer:
            pointer_logits = pointer_mask.apply_monotonic_pointer_mask_tf(
                pointer_logits,
                prev_patches,
            )
        pointer_logits = pointer_mask.apply_soft_distance_prior_tf(
            pointer_logits,
            prev_patches,
            alpha=soft_alpha,
        )
    elif pointer_logits is not None and monotonic_pointer:
        pointer_logits = pointer_mask.apply_monotonic_pointer_mask_tf(
            pointer_logits,
            prev_patches,
        )
    if gap_alignment and gap_logits is not None:
        if gap_delta_lookup is None:
            raise ValueError("gap_alignment requires gap_delta_lookup")
        gap_logits = pointer_mask.apply_gap_soft_distance_prior_tf(
            gap_logits,
            prev_patches,
            alpha=soft_alpha,
            delta_lookup=gap_delta_lookup,
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

    onset_count = tf.reduce_sum(onset_step_mask) + 1e-9
    gap_loss = tf.constant(0.0, dtype=tf.float32)
    pointer_loss = tf.constant(0.0, dtype=tf.float32)
    alignment_term = tf.constant(0.0, dtype=tf.float32)

    if gap_alignment:
        if gap_logits is None:
            raise ValueError("gap_alignment requires outputs['gap_logits']")
        if gap_delta_lookup is None:
            raise ValueError("gap_alignment requires gap_delta_lookup")
        target_gap_ids = tf.cast(batch["target_gap_ids"], tf.int32)
        gap_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_gap_ids,
            logits=gap_logits,
        )
        gap_losses = gap_losses * onset_step_mask
        gap_loss = tf.reduce_sum(gap_losses) / onset_count
        alignment_term = tf.cast(gap_loss_weight, tf.float32) * gap_loss
        # Report primary alignment CE under pointer_loss for checkpoint continuity.
        pointer_loss = gap_loss
        if pointer_logits is not None and float(pointer_loss_weight) > 0.0:
            abs_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_patch_indices,
                logits=pointer_logits,
            )
            abs_losses = abs_losses * onset_step_mask
            abs_pointer_loss = tf.reduce_sum(abs_losses) / onset_count
            alignment_term = (
                alignment_term
                + tf.cast(pointer_loss_weight, tf.float32) * abs_pointer_loss
            )
        n_patches = tf.shape(batch["patch_mask"])[1]
        max_patch = n_patches - 1
        resolved = resolved_patches_from_gap_logits(
            gap_logits,
            prev_patches,
            delta_lookup=gap_delta_lookup,
            max_patch=max_patch,
        )
        pred_times = predicted_times_from_patches(
            resolved,
            residual_sec,
            patch_frames=patch_frames,
            hop_sec=hop_sec,
        )
        hard_patch = resolved
    else:
        if pointer_logits is None:
            raise ValueError("pointer alignment requires outputs['pointer_logits']")
        anchor = str(pointer_local_ce_anchor or "target").lower()
        radius = int(pointer_local_ce_radius)
        pointer_ce_mask = onset_step_mask
        if radius > 0 and anchor == "prev":
            pointer_logits = pointer_mask.apply_prev_relative_window_tf(
                pointer_logits,
                prev_patches,
                max_ahead=radius,
            )
            pointer_logits_for_ce = pointer_logits
            pointer_ce_mask = pointer_mask.prev_relative_ce_step_mask(
                target_patch_indices,
                prev_patches,
                max_ahead=radius,
                onset_step_mask=onset_step_mask,
            )
        else:
            pointer_logits_for_ce = apply_local_pointer_ce_mask(
                pointer_logits,
                target_patch_indices,
                radius=radius,
            )
        pointer_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_patch_indices,
            logits=pointer_logits_for_ce,
        )
        pointer_losses = pointer_losses * pointer_ce_mask
        pointer_ce_count = tf.reduce_sum(pointer_ce_mask) + 1e-9
        pointer_loss = tf.reduce_sum(pointer_losses) / pointer_ce_count
        alignment_term = tf.cast(pointer_loss_weight, tf.float32) * pointer_loss
        pred_times = predicted_times_from_outputs(
            pointer_logits,
            residual_sec,
            patch_frames=patch_frames,
            hop_sec=hop_sec,
            use_soft_expected=use_soft_pointer_time and not use_ste_pointer_time,
            use_ste=use_ste_pointer_time,
        )
        hard_patch = tf.argmax(pointer_logits, axis=-1, output_type=tf.int32)

    time_mask = onset_step_mask
    if time_loss_correct_patch_only:
        correct = tf.cast(
            tf.equal(hard_patch, target_patch_indices),
            tf.float32,
        )
        time_mask = onset_step_mask * correct
    time_errors = tf.abs(pred_times - target_times) * time_mask
    time_denom = tf.reduce_sum(time_mask) + 1e-9
    time_loss = tf.reduce_sum(time_errors) / time_denom

    residual_sq = tf.square(residual_sec - target_residual_sec) * onset_step_mask
    residual_loss = tf.reduce_sum(residual_sq) / onset_count

    total_loss = (
        token_loss
        + alignment_term
        + tf.cast(lambda_time, tf.float32) * time_loss
        + tf.cast(lambda_residual, tf.float32) * residual_loss
    )
    return total_loss, {
        "token_loss": token_loss,
        "pointer_loss": pointer_loss,
        "gap_loss": gap_loss,
        "time_loss": time_loss,
        "residual_loss": residual_loss,
    }


def masked_gap_accuracy(
    gap_logits: tf.Tensor,
    target_gap_ids: tf.Tensor,
    onset_step_mask: tf.Tensor,
) -> tf.Tensor:
    """Top-1 gap-id accuracy over onset decoder steps."""
    predictions = tf.argmax(gap_logits, axis=-1, output_type=tf.int32)
    target_gap_ids = tf.cast(target_gap_ids, tf.int32)
    correct = tf.cast(tf.equal(predictions, target_gap_ids), tf.float32)
    mask = tf.cast(onset_step_mask, tf.float32)
    return tf.reduce_sum(correct * mask) / (tf.reduce_sum(mask) + 1e-9)


def masked_pointer_patch_accuracy(
    pointer_logits: tf.Tensor,
    target_patch_indices: tf.Tensor,
    onset_step_mask: tf.Tensor,
) -> tf.Tensor:
    """Top-1 patch accuracy over onset decoder steps.

    ``pointer_logits`` should already include the same monotonic mask used for
    pointer CE when ``monotonic_pointer`` is on (call after that mask).
    """
    predictions = tf.argmax(pointer_logits, axis=-1, output_type=tf.int32)
    target_patch_indices = tf.cast(target_patch_indices, tf.int32)
    correct = tf.cast(tf.equal(predictions, target_patch_indices), tf.float32)
    mask = tf.cast(onset_step_mask, tf.float32)
    return tf.reduce_sum(correct * mask) / (tf.reduce_sum(mask) + 1e-9)


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
