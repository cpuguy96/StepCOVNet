"""Masked grid-weighted placement, auxiliary beat, and R-Drop losses for ITGPT."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet.itgpt import constants


def grid_importance_weights(
    n_slots: int = constants.N_SLOTS,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the ITGPT 48-slot BCE weights.

    Args:
        n_slots: Slot count (paper: 48).
        device: Optional tensor device.

    Returns:
        Float32 tensor of shape ``(n_slots,)``.
    """
    weights = [constants.GRID_WEIGHT_MICRO] * n_slots
    for index in constants.INDICES_16TH:
        weights[index] = constants.GRID_WEIGHT_16TH
    for index in constants.INDICES_24TH:
        weights[index] = constants.GRID_WEIGHT_24TH
    for index in constants.INDICES_32ND:
        weights[index] = 1.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_beat_aux_loss(
    beat_active_logit: torch.Tensor,
    beat_count_pred: torch.Tensor,
    targets: torch.Tensor,
    beat_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked auxiliary beat-activity BCE + log-count SmoothL1 loss.

    Args:
        beat_active_logit: Per-beat activity logits ``(B, T)``.
        beat_count_pred: Per-beat ``log1p(step_count)`` predictions ``(B, T)``.
        targets: Binary slot targets ``(B, T, 48)``.
        beat_mask: Real-beat mask ``(B, T)`` with 1 on real beats.

    Returns:
        Scalar float32 auxiliary loss averaged over valid beats.
    """
    targets_fp32 = targets.float()
    mask_fp32 = beat_mask.float()
    denom = torch.clamp(mask_fp32.sum(), min=1.0)
    active_target = (targets_fp32.amax(dim=-1) >= 0.5).float()
    active_bce = nn_func.binary_cross_entropy_with_logits(
        beat_active_logit.float(),
        active_target,
        reduction="none",
    )
    count_target = torch.log1p(targets_fp32.sum(dim=-1))
    count_l1 = nn_func.smooth_l1_loss(
        beat_count_pred.float(),
        count_target,
        reduction="none",
    )
    per_beat = 0.5 * (active_bce + count_l1)
    return (per_beat * mask_fp32).sum() / denom


class ItgptPlacementLoss(nn.Module):
    """Masked grid-weighted slot BCE-with-logits with optional smoothing and Dice.

    Takes raw float32 logits for numerical stability under mixed precision.

    Args:
        label_smoothing: Binary label smoothing factor in ``[0, 1)`` mapping
            targets ``y -> y * (1 - eps) + 0.5 * eps`` (default ``0.0``).
        dice_loss_weight: Non-negative weight for an additive masked soft-Dice
            term computed on ``sigmoid(logits)`` and raw binary targets over
            real beats (``0.0`` disables).
        bce_pos_weight: Positive class weight passed to
            ``binary_cross_entropy_with_logits`` (``1.0`` disables).
        normalize_by_valid_beats: If True, divide masked beat losses by the
            number of real beats (``mask.sum()``). If False, divide by the
            total padded beat count (mean over batch and beat axes).

    Raises:
        ValueError: If ``label_smoothing`` is outside ``[0, 1)``,
            ``dice_loss_weight`` is negative, or ``bce_pos_weight`` is not
            positive.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        dice_loss_weight: float = 0.0,
        bce_pos_weight: float = 1.0,
        *,
        normalize_by_valid_beats: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        if float(dice_loss_weight) < 0.0:
            raise ValueError(f"dice_loss_weight must be >= 0, got {dice_loss_weight}")
        if float(bce_pos_weight) <= 0.0:
            raise ValueError(f"bce_pos_weight must be positive, got {bce_pos_weight}")
        self.label_smoothing = float(label_smoothing)
        self.dice_loss_weight = float(dice_loss_weight)
        self.bce_pos_weight = float(bce_pos_weight)
        self.normalize_by_valid_beats = bool(normalize_by_valid_beats)
        self.register_buffer(
            "grid_weights",
            grid_importance_weights(),
            persistent=False,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        beat_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar masked slot-weighted BCE loss plus optional soft Dice.

        Args:
            logits: Raw logits ``(batch, beats, 48)``.
            targets: Binary slot targets ``(batch, beats, 48)``.
            beat_mask: Real-beat mask ``(batch, beats)`` with 1 on real beats.

        Returns:
            Scalar float32 loss tensor.
        """
        logits_fp32 = logits.float()
        raw_targets_fp32 = targets.float()
        targets_fp32 = raw_targets_fp32
        if self.label_smoothing > 0.0:
            targets_fp32 = (
                raw_targets_fp32 * (1.0 - self.label_smoothing)
                + 0.5 * self.label_smoothing
            )
        mask_fp32 = beat_mask.float()
        pos_weight = (
            torch.tensor(
                self.bce_pos_weight, dtype=torch.float32, device=logits_fp32.device
            )
            if self.bce_pos_weight != 1.0
            else None
        )
        per_slot = nn_func.binary_cross_entropy_with_logits(
            logits_fp32,
            targets_fp32,
            pos_weight=pos_weight,
            reduction="none",
        )
        grid_weights: torch.Tensor = self.grid_weights  # type: ignore[assignment]
        per_beat = torch.mean(per_slot * grid_weights.to(logits_fp32.device), dim=-1)
        masked = per_beat * mask_fp32
        if self.normalize_by_valid_beats:
            denom = torch.clamp(mask_fp32.sum(), min=1.0)
            bce_loss = masked.sum() / denom
        else:
            bce_loss = masked.mean()
        if self.dice_loss_weight == 0.0:
            return bce_loss
        dice_probs = torch.sigmoid(logits_fp32)
        slot_mask = mask_fp32.unsqueeze(-1)
        masked_probs = dice_probs * slot_mask
        masked_targets = raw_targets_fp32 * slot_mask
        eps = 1e-6
        intersection = torch.sum(masked_probs * masked_targets)
        dice_denom = torch.sum(masked_probs) + torch.sum(masked_targets) + eps
        dice_loss = 1.0 - (2.0 * intersection + eps) / dice_denom
        return bce_loss + self.dice_loss_weight * dice_loss


def compute_bernoulli_rdrop_kl(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    beat_mask: torch.Tensor,
    grid_weights: torch.Tensor | None = None,
    *,
    normalize_by_valid_beats: bool = False,
) -> torch.Tensor:
    """Compute masked grid-weighted bidirectional Bernoulli KL for R-Drop.

    Args:
        logits1: First stochastic pass logits ``(batch, beats, n_slots)``.
        logits2: Second stochastic pass logits ``(batch, beats, n_slots)``.
        beat_mask: Real-beat mask ``(batch, beats)`` with 1 on real beats.
        grid_weights: Optional per-slot importance weights ``(n_slots,)``.
            Defaults to :func:`grid_importance_weights` on ``logits1``'s device.
        normalize_by_valid_beats: If True, average over valid beats
            (``beat_mask.sum()``); otherwise average over all padded beats to
            match :class:`ItgptPlacementLoss` default scaling.

    Returns:
        Scalar float32 symmetric KL loss tensor.
    """
    z1 = logits1.float()
    z2 = logits2.float()
    p1 = torch.sigmoid(z1)
    p2 = torch.sigmoid(z2)
    log_p1 = nn_func.logsigmoid(z1)
    log_1mp1 = nn_func.logsigmoid(-z1)
    log_p2 = nn_func.logsigmoid(z2)
    log_1mp2 = nn_func.logsigmoid(-z2)
    kl_12 = p1 * (log_p1 - log_p2) + (1.0 - p1) * (log_1mp1 - log_1mp2)
    kl_21 = p2 * (log_p2 - log_p1) + (1.0 - p2) * (log_1mp2 - log_1mp1)
    sym_kl = 0.5 * (kl_12 + kl_21)
    weights = (
        grid_importance_weights(device=z1.device)
        if grid_weights is None
        else grid_weights.to(device=z1.device, dtype=torch.float32)
    )
    per_beat = torch.mean(sym_kl * weights, dim=-1)
    masked = per_beat * beat_mask.float()
    if normalize_by_valid_beats:
        denom = torch.clamp(beat_mask.float().sum(), min=1.0)
        return masked.sum() / denom
    return masked.mean()
