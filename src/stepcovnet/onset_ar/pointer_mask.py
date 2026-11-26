"""Monotonic pointer masking and soft-distance priors (PyTorch & NumPy)."""

from __future__ import annotations

import numpy as np

from stepcovnet.pytorch.onset_ar.losses import (
    apply_gap_soft_distance_prior,
    apply_monotonic_pointer_mask,
    apply_prev_relative_window,
    apply_soft_distance_prior,
    prev_relative_ce_step_mask,
    teacher_forced_prev_patch_indices,
)


def apply_monotonic_pointer_mask_numpy(
    pointer_logits: np.ndarray,
    prev_patch: int,
) -> np.ndarray:
    """Mask a single-step logit vector below ``prev_patch``.

    Args:
        pointer_logits: Pointer logit array.
        prev_patch: Minimum allowed patch index.

    Returns:
        Masked float32 copy.
    """
    logits = np.asarray(pointer_logits, dtype=np.float32).copy()
    prev = max(0, int(prev_patch))
    if prev > 0:
        logits[..., :prev] = -1e9
    return logits


def apply_prev_relative_window_numpy(
    pointer_logits: np.ndarray,
    prev_patch: int,
    *,
    max_ahead: int,
) -> np.ndarray:
    """Mask logits strictly beyond ``prev_patch + max_ahead`` when ``prev_patch > 0``.

    Args:
        pointer_logits: Pointer logit array.
        prev_patch: Previous patch index.
        max_ahead: Inclusive forward window width.

    Returns:
        Masked float32 array.
    """
    if max_ahead <= 0 or int(prev_patch) <= 0:
        return np.asarray(pointer_logits, dtype=np.float32)
    logits = np.asarray(pointer_logits, dtype=np.float32).copy()
    hi = int(prev_patch) + int(max_ahead)
    if hi + 1 < logits.shape[-1]:
        logits[..., hi + 1 :] = -1e9
    return logits


def apply_soft_distance_prior_numpy(
    pointer_logits: np.ndarray,
    prev_patch: int,
    *,
    alpha: float,
) -> np.ndarray:
    """Subtract ``alpha * max(0, p - prev_patch)`` when ``prev_patch > 0``.

    Args:
        pointer_logits: Pointer logit array.
        prev_patch: Previous patch index.
        alpha: Non-negative distance slope.

    Returns:
        Adjusted float32 array.
    """
    if float(alpha) <= 0.0 or int(prev_patch) <= 0:
        return np.asarray(pointer_logits, dtype=np.float32)
    logits = np.asarray(pointer_logits, dtype=np.float32).copy()
    ahead = np.maximum(
        np.arange(logits.shape[-1], dtype=np.float32) - float(prev_patch), 0.0
    )
    return logits - float(alpha) * ahead


def teacher_forced_prev_patch_indices_numpy(
    target_patch_indices: np.ndarray,
) -> np.ndarray:
    """Shift target patch indices right by 1 with 0 at step 0.

    Args:
        target_patch_indices: 1D or 2D int array of target patch indices.

    Returns:
        Int32 array of previous patch indices.
    """
    arr = np.asarray(target_patch_indices, dtype=np.int32)
    out = np.zeros_like(arr)
    if arr.ndim == 1:
        if arr.size > 1:
            out[1:] = arr[:-1]
    elif arr.shape[1] > 1:
        out[:, 1:] = arr[:, :-1]
    return out


__all__ = [
    "apply_gap_soft_distance_prior",
    "apply_monotonic_pointer_mask",
    "apply_monotonic_pointer_mask_numpy",
    "apply_prev_relative_window",
    "apply_prev_relative_window_numpy",
    "apply_soft_distance_prior",
    "apply_soft_distance_prior_numpy",
    "prev_relative_ce_step_mask",
    "teacher_forced_prev_patch_indices",
    "teacher_forced_prev_patch_indices_numpy",
]
