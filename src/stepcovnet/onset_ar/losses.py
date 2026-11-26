"""AR onset losses and masked accuracy metrics (PyTorch)."""

from __future__ import annotations

from stepcovnet.pytorch.onset_ar.losses import (
    apply_gap_soft_distance_prior,
    apply_local_pointer_ce_mask,
    apply_monotonic_pointer_mask,
    apply_prev_relative_window,
    apply_soft_distance_prior,
    build_token_class_weights_numpy,
    compute_ar_onset_loss,
    masked_alignment_accuracy,
    masked_token_accuracy,
    predicted_times_from_outputs,
    prev_relative_ce_step_mask,
    teacher_forced_prev_patch_indices,
)

__all__ = [
    "apply_gap_soft_distance_prior",
    "apply_local_pointer_ce_mask",
    "apply_monotonic_pointer_mask",
    "apply_prev_relative_window",
    "apply_soft_distance_prior",
    "build_token_class_weights_numpy",
    "compute_ar_onset_loss",
    "masked_alignment_accuracy",
    "masked_token_accuracy",
    "predicted_times_from_outputs",
    "prev_relative_ce_step_mask",
    "teacher_forced_prev_patch_indices",
]
