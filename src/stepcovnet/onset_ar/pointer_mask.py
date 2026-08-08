"""Monotonic pointer masking helpers for train and decode."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def teacher_forced_prev_patch_indices(
    target_patch_indices: tf.Tensor,
) -> tf.Tensor:
    """Previous onset patch per decoder step (0 at the first step)."""
    target_patch_indices = tf.cast(target_patch_indices, tf.int32)
    return tf.pad(target_patch_indices[:, :-1], [[0, 0], [1, 0]])


def apply_monotonic_pointer_mask_tf(
    pointer_logits: tf.Tensor,
    prev_patch_indices: tf.Tensor,
) -> tf.Tensor:
    """Mask logits for patches strictly before ``prev_patch_indices``."""
    pointer_logits = tf.cast(pointer_logits, tf.float32)
    n_patches = tf.shape(pointer_logits)[-1]
    patch_ids = tf.range(n_patches, dtype=tf.int32)
    patch_ids = tf.reshape(patch_ids, (1, 1, n_patches))
    prev = tf.expand_dims(tf.cast(prev_patch_indices, tf.int32), axis=-1)
    invalid = tf.cast(patch_ids < prev, pointer_logits.dtype)
    return pointer_logits + invalid * (-1e9)


def apply_monotonic_pointer_mask_numpy(
    pointer_logits: np.ndarray,
    prev_patch: int,
) -> np.ndarray:
    """Mask a single-step logit vector below ``prev_patch``."""
    logits = np.asarray(pointer_logits, dtype=np.float32).copy()
    prev_patch = max(0, int(prev_patch))
    if prev_patch > 0:
        logits[..., :prev_patch] = -1e9
    return logits


def apply_prev_relative_window_tf(
    pointer_logits: tf.Tensor,
    prev_patch_indices: tf.Tensor,
    *,
    max_ahead: int,
) -> tf.Tensor:
    """Mask patches strictly after ``prev + max_ahead`` (decode-consistent local).

    Combined with :func:`apply_monotonic_pointer_mask_tf`, this yields the
    allowed set ``[prev, prev + max_ahead]`` when ``prev > 0``. When
    ``prev == 0`` (first onset), no upper bound is applied — songs often start
    well after patch ``max_ahead``, and masking those targets poisons CE.
    ``max_ahead <= 0`` is a no-op.

    Rare teacher gaps larger than ``max_ahead`` can still leave the *label*
    masked; callers should zero those steps via
    :func:`prev_relative_ce_step_mask`.
    """
    if max_ahead <= 0:
        return pointer_logits
    pointer_logits = tf.cast(pointer_logits, tf.float32)
    n_patches = tf.shape(pointer_logits)[-1]
    patch_ids = tf.range(n_patches, dtype=tf.int32)
    patch_ids = tf.reshape(patch_ids, (1, 1, n_patches))
    prev = tf.expand_dims(tf.cast(prev_patch_indices, tf.int32), axis=-1)
    hi = prev + int(max_ahead)
    apply_upper = prev > 0
    invalid = tf.cast(
        tf.logical_and(patch_ids > hi, apply_upper),
        pointer_logits.dtype,
    )
    return pointer_logits + invalid * (-1e9)


def prev_relative_ce_step_mask(
    target_patch_indices: tf.Tensor,
    prev_patch_indices: tf.Tensor,
    *,
    max_ahead: int,
    onset_step_mask: tf.Tensor,
) -> tf.Tensor:
    """Onset-step mask that drops targets outside ``[prev, prev+max_ahead]``.

    First-onset steps (``prev == 0``) stay active — matching the unrestricted
    upper bound in :func:`apply_prev_relative_window_tf`. Used so sparse CE
    never sees a label sitting on a ``-1e9`` logit (section gaps > R).
    """
    onset = tf.cast(onset_step_mask, tf.float32)
    if max_ahead <= 0:
        return onset
    target = tf.cast(target_patch_indices, tf.int32)
    prev = tf.cast(prev_patch_indices, tf.int32)
    in_window = tf.logical_or(
        prev <= 0,
        target <= prev + int(max_ahead),
    )
    return onset * tf.cast(in_window, tf.float32)


def apply_prev_relative_window_numpy(
    pointer_logits: np.ndarray,
    prev_patch: int,
    *,
    max_ahead: int,
) -> np.ndarray:
    """Numpy version of :func:`apply_prev_relative_window_tf` for one step."""
    if max_ahead <= 0 or int(prev_patch) <= 0:
        return pointer_logits
    logits = np.asarray(pointer_logits, dtype=np.float32).copy()
    hi = int(prev_patch) + int(max_ahead)
    if hi + 1 < logits.shape[-1]:
        logits[..., hi + 1 :] = -1e9
    return logits


def apply_soft_distance_prior_tf(
    pointer_logits: tf.Tensor,
    prev_patch_indices: tf.Tensor,
    *,
    alpha: float | tf.Tensor,
) -> tf.Tensor:
    """Subtract ``alpha * max(0, patch - prev)`` from logits (no hard cutoff).

    Encourages mass near ``prev`` while keeping long jumps reachable. ``alpha``
    may be a ``tf.Variable`` (anneal); ``alpha == 0`` is a no-op. Skipped when
    ``prev == 0`` (first onset may land far into the song). Combine with
    :func:`apply_monotonic_pointer_mask_tf` for ``p >= prev``.
    """
    pointer_logits = tf.cast(pointer_logits, tf.float32)
    alpha_t = tf.cast(alpha, tf.float32)
    n_patches = tf.shape(pointer_logits)[-1]
    patch_ids = tf.range(n_patches, dtype=tf.float32)
    patch_ids = tf.reshape(patch_ids, (1, 1, n_patches))
    prev = tf.cast(prev_patch_indices, tf.float32)
    prev = tf.expand_dims(prev, axis=-1)
    ahead = tf.nn.relu(patch_ids - prev)
    apply = tf.cast(prev > 0.0, pointer_logits.dtype)
    return pointer_logits - (alpha_t * ahead * apply)


def apply_soft_distance_prior_numpy(
    pointer_logits: np.ndarray,
    prev_patch: int,
    *,
    alpha: float,
) -> np.ndarray:
    """Numpy version of :func:`apply_soft_distance_prior_tf` for one step."""
    if float(alpha) <= 0.0 or int(prev_patch) <= 0:
        return pointer_logits
    logits = np.asarray(pointer_logits, dtype=np.float32).copy()
    n = logits.shape[-1]
    ahead = np.arange(n, dtype=np.float32) - float(prev_patch)
    ahead = np.maximum(ahead, 0.0)
    logits = logits - float(alpha) * ahead
    return logits


def apply_gap_soft_distance_prior_tf(
    gap_logits: tf.Tensor,
    prev_patch_indices: tf.Tensor,
    *,
    alpha: float | tf.Tensor,
    delta_lookup: tf.Tensor,
) -> tf.Tensor:
    """Subtract ``alpha * decode_delta(id)`` from gap logits (no hard cutoff).

    Encourages small Δ while keeping long jumps reachable. ``alpha`` may be a
    ``tf.Variable`` (anneal); ``alpha == 0`` is a no-op. Skipped when
    ``prev == 0`` (first onset may need a large absolute Δ).
    ``delta_lookup[id]`` is ``PatchGapVocab.decode_delta(id)``.
    """
    gap_logits = tf.cast(gap_logits, tf.float32)
    alpha_t = tf.cast(alpha, tf.float32)
    deltas = tf.cast(delta_lookup, tf.float32)
    penalty = alpha_t * tf.reshape(deltas, (1, 1, -1))
    prev = tf.cast(prev_patch_indices, tf.float32)
    apply = tf.cast(tf.expand_dims(prev, axis=-1) > 0.0, gap_logits.dtype)
    return gap_logits - penalty * apply


def apply_gap_soft_distance_prior_numpy(
    gap_logits: np.ndarray,
    prev_patch: int,
    *,
    alpha: float,
    delta_lookup: np.ndarray,
) -> np.ndarray:
    """Numpy version of :func:`apply_gap_soft_distance_prior_tf` for one step."""
    if float(alpha) <= 0.0 or int(prev_patch) <= 0:
        return gap_logits
    logits = np.asarray(gap_logits, dtype=np.float32).copy()
    deltas = np.asarray(delta_lookup, dtype=np.float32)
    return logits - float(alpha) * deltas


def teacher_forced_prev_patch_indices_numpy(
    target_patch_indices: np.ndarray,
) -> np.ndarray:
    """Numpy version of :func:`teacher_forced_prev_patch_indices`."""
    target_patch_indices = np.asarray(target_patch_indices, dtype=np.int32)
    if target_patch_indices.ndim == 1:
        out = np.zeros_like(target_patch_indices)
        out[1:] = target_patch_indices[:-1]
        return out
    out = np.zeros_like(target_patch_indices)
    out[:, 1:] = target_patch_indices[:, :-1]
    return out
