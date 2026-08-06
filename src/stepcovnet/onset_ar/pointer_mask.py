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
