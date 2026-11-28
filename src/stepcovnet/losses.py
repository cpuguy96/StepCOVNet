"""PyTorch dense onset and arrow training loss functions."""

from __future__ import annotations

from stepcovnet.core.losses import (
    arrow_label_smoothed_crossentropy,
    build_arrow_combined_loss,
    chart_validity_auxiliary_loss,
    dense_onset_bce_with_logits_loss,
    masked_mse_aux_interval,
    note_kind_balance_auxiliary_loss,
    sparse_focal_loss,
)

__all__ = [
    "arrow_label_smoothed_crossentropy",
    "build_arrow_combined_loss",
    "chart_validity_auxiliary_loss",
    "dense_onset_bce_with_logits_loss",
    "masked_mse_aux_interval",
    "note_kind_balance_auxiliary_loss",
    "sparse_focal_loss",
]
