"""Tests for monotonic pointer masking."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import pointer_mask


def test_teacher_forced_prev_patch_indices_shifts() -> None:
    targets = tf.constant([[3, 5, 8, 8]], dtype=tf.int32)
    prev = pointer_mask.teacher_forced_prev_patch_indices(targets)
    np.testing.assert_array_equal(prev.numpy(), [[0, 3, 5, 8]])


def test_monotonic_mask_blocks_earlier_patches() -> None:
    logits = tf.constant([[[1.0, 2.0, 3.0, 4.0]]], dtype=tf.float32)
    prev = tf.constant([[2]], dtype=tf.int32)
    masked = pointer_mask.apply_monotonic_pointer_mask_tf(logits, prev)
    values = masked.numpy()[0, 0]
    assert values[0] < -1e8
    assert values[1] < -1e8
    assert values[2] > 0
    assert values[3] > 0


def test_monotonic_mask_numpy_single_step() -> None:
    logits = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    masked = pointer_mask.apply_monotonic_pointer_mask_numpy(logits, 2)
    assert masked[0] < -1e8
    assert masked[1] < -1e8
    assert masked[2] == 3.0
