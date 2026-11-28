"""PyTorch dense onset and arrow losses (focal, label smoothing, chart validity, diversity)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional as nn_func

from stepcovnet import config, constants


def _build_arrow_digit_onehot_table() -> torch.Tensor:
    table = np.zeros((constants.N_ARROW_TYPES, 4, 4), dtype=np.float32)
    for n in range(constants.N_ARROW_TYPES):
        table[n, 0, (n // 64) % 4] = 1.0
        table[n, 1, (n // 16) % 4] = 1.0
        table[n, 2, (n // 4) % 4] = 1.0
        table[n, 3, n % 4] = 1.0
    return torch.from_numpy(table)


_ARROW_DIGIT_ONEHOT_TABLE = _build_arrow_digit_onehot_table()


def dense_onset_bce_with_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute optionally weighted/masked BCE-with-logits for dense onset models.

    Args:
        logits: Per-frame onset logits ``(B, T, 1)`` or ``(B, T)``.
        targets: Binary/Gaussian frame targets matching ``logits`` shape.
        pos_weight: Positive class weight multiplier.
        mask: Optional float validity mask broadcastable to ``logits``.

    Returns:
        Scalar float32 loss tensor.
    """
    logits_f = logits.float().reshape(targets.shape)
    targets_f = targets.float()
    pw = torch.tensor(float(pos_weight), device=logits_f.device, dtype=torch.float32)
    per_elem = nn_func.binary_cross_entropy_with_logits(
        logits_f, targets_f, pos_weight=pw, reduction="none"
    )
    if mask is None:
        return per_elem.mean()
    m = mask.float().reshape(per_elem.shape)
    return (per_elem * m).sum() / torch.clamp(m.sum(), min=1.0)


def sparse_focal_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    gamma: float,
    *,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> torch.Tensor:
    """Sparse categorical focal loss over arrow probability predictions.

    Args:
        y_true: Target class indices ``(B, S)``.
        y_pred: Class probabilities ``(B, S, 256)``.
        gamma: Focusing exponent.
        ignore_class: Padding class index excluded from loss.

    Returns:
        Scalar mean focal loss over valid positions.
    """
    targets_i = y_true.to(torch.long).clamp(0, constants.N_ARROW_TYPES - 1)
    p_t = torch.gather(y_pred.float(), dim=-1, index=targets_i.unsqueeze(-1)).squeeze(
        -1
    )
    p_t = p_t.clamp(1e-7, 1.0 - 1e-7)
    loss_per_step = ((1.0 - p_t) ** float(gamma)) * (-torch.log(p_t))
    mask = (y_true != ignore_class).float()
    return (loss_per_step * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def arrow_label_smoothed_crossentropy(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smoothing: float,
    *,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> torch.Tensor:
    """Label-smoothed cross-entropy over arrow probability predictions.

    Args:
        y_true: Target class indices ``(B, S)``.
        y_pred: Class probabilities ``(B, S, 256)``.
        smoothing: Label smoothing factor in ``[0, 1)``.
        ignore_class: Padding class index excluded from loss.

    Returns:
        Scalar mean cross-entropy over valid positions.
    """
    targets_i = y_true.to(torch.long).clamp(0, constants.N_ARROW_TYPES - 1)
    one_hot = nn_func.one_hot(targets_i, num_classes=constants.N_ARROW_TYPES).float()
    smoothed = one_hot * (1.0 - float(smoothing)) + float(smoothing) / float(
        constants.N_ARROW_TYPES
    )
    probs = y_pred.float().clamp(1e-7, 1.0)
    per_step = -(smoothed * torch.log(probs)).sum(dim=-1)
    mask = (y_true != ignore_class).float()
    return (per_step * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_mse_aux_interval(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked MSE for auxiliary next-interval regression.

    Args:
        y_true: Target intervals ``(B, S, 1)``.
        y_pred: Predicted intervals ``(B, S, 1)``.
        sample_weight: Optional validity mask ``(B, S, 1)``.

    Returns:
        Scalar float32 MSE.
    """
    sq = torch.square(y_pred.float() - y_true.float())
    if sample_weight is None:
        return sq.mean()
    w = sample_weight.float()
    return (sq * w).sum() / torch.clamp(w.sum(), min=1.0)


def _chart_validity_violation_weights(
    prob_digit: torch.Tensor,
    last_valid_weights: torch.Tensor,
    *,
    soft: bool = True,
) -> torch.Tensor:
    p1 = prob_digit[:, :, :, 1]
    p2 = prob_digit[:, :, :, 2]
    p3 = prob_digit[:, :, :, 3]
    run_soft = torch.cumsum(p2 - p3, dim=1)
    run_prev = torch.cat(
        [torch.zeros_like(run_soft[:, :1, :]), run_soft[:, :-1, :]], dim=1
    )
    penalty_orphan_3 = p3 * torch.clamp(1.0 - run_prev, min=0.0)
    state_holding = (
        torch.sigmoid((run_prev - 0.5) * 10.0) if soft else (run_prev >= 1.0).float()
    )
    penalty_tap_hold = p1 * state_holding
    penalty_nested = p2 * state_holding
    penalty_unterm = last_valid_weights * torch.clamp(run_soft, min=0.0)
    return penalty_orphan_3 + penalty_tap_hold + penalty_nested + penalty_unterm


def chart_validity_auxiliary_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> torch.Tensor:
    """Differentiable 2-state FREE/HOLDING StepMania chart validity penalty.

    Args:
        y_true: Target arrow codes ``(B, S)``.
        y_pred: Predicted arrow probabilities ``(B, S, 256)``.
        ignore_class: Padding class excluded from loss.

    Returns:
        Scalar mean violation penalty over valid step-columns.
    """
    device = y_pred.device
    table = _ARROW_DIGIT_ONEHOT_TABLE.to(device=device, dtype=torch.float32)
    prob_digit = torch.einsum("bsn,ncd->bscd", y_pred.float(), table)
    mask_bool = y_true != ignore_class
    mask = mask_bool.float()
    prob_digit = prob_digit * mask.unsqueeze(-1).unsqueeze(-1)

    b, seq_len = int(y_true.shape[0]), int(y_true.shape[1])
    seq_idx = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    last_valid_idx = torch.argmax(mask_bool.to(torch.long) * seq_idx, dim=1)
    last_valid_weights = torch.zeros(
        (b, seq_len, 4), device=device, dtype=torch.float32
    )
    valid_batch = mask.sum(dim=1) > 0.0
    batch_ids = torch.arange(b, device=device, dtype=torch.long)
    last_valid_weights[batch_ids, last_valid_idx, :] = valid_batch.float().unsqueeze(-1)

    weights = _chart_validity_violation_weights(
        prob_digit, last_valid_weights, soft=True
    )
    step_masked = (weights * mask.unsqueeze(-1)).sum()
    return step_masked / (torch.clamp(mask.sum(), min=1.0) * 4.0)


def note_kind_balance_auxiliary_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    ignore_class: int = constants.ARROW_PADDING_CLASS,
) -> torch.Tensor:
    """Differentiable MSE aligning predicted hold-rate with ground-truth hold-rate.

    Args:
        y_true: Target arrow codes ``(B, S)``.
        y_pred: Predicted arrow probabilities ``(B, S, 256)``.
        ignore_class: Padding class excluded from loss.

    Returns:
        Scalar MSE over valid steps.
    """
    table = _ARROW_DIGIT_ONEHOT_TABLE.to(device=y_pred.device, dtype=torch.float32)
    pred_digit = torch.einsum("bsn,ncd->bscd", y_pred.float(), table)
    pred_hold = (pred_digit[:, :, :, 2] + pred_digit[:, :, :, 3]).mean(dim=-1)

    targets_i = y_true.to(torch.long).clamp(0, constants.N_ARROW_TYPES - 1)
    true_onehot = nn_func.one_hot(
        targets_i, num_classes=constants.N_ARROW_TYPES
    ).float()
    true_digit = torch.einsum("bsn,ncd->bscd", true_onehot, table)
    true_hold = (true_digit[:, :, :, 2] + true_digit[:, :, :, 3]).mean(dim=-1)

    mask = (y_true != ignore_class).float()
    return (torch.square(pred_hold - true_hold) * mask).sum() / torch.clamp(
        mask.sum(), min=1.0
    )


def build_arrow_combined_loss(
    run_config: config.ArrowRunConfig,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build combined PyTorch arrow loss matching ``stepcovnet.losses.build_arrow_combined_loss``.

    Args:
        run_config: Arrow run configuration.

    Returns:
        Callable ``(y_true, y_pred_probs) -> scalar Tensor``.
    """
    w_val = float(run_config.chart_validity_aux_weight)
    w_div = float(run_config.diversity_aux_weight)
    rej_thr = run_config.chart_validity_rejection_threshold
    rej_scale = float(run_config.chart_validity_rejection_scale)
    rej_temp = float(run_config.chart_validity_rejection_temperature)

    def _main(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if run_config.loss_type == "crossentropy":
            return arrow_label_smoothed_crossentropy(
                y_true,
                y_pred,
                smoothing=float(run_config.label_smoothing),
            )
        return sparse_focal_loss(y_true, y_pred, gamma=float(run_config.focal_gamma))

    def _combined(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        main = _main(y_true, y_pred)
        validity = chart_validity_auxiliary_loss(y_true, y_pred)
        diversity = note_kind_balance_auxiliary_loss(y_true, y_pred)
        if rej_thr is None:
            return main + validity * w_val + diversity * w_div
        validity_score = 1.0 - validity
        gate = torch.sigmoid((validity_score - float(rej_thr)) * rej_temp)
        return gate * (main + diversity * w_div) + (1.0 - gate) * rej_scale * validity

    return _combined
