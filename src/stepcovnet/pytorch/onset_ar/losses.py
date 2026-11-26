"""PyTorch losses, pointer masks, and metrics for AR onset training."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as nn_func

from stepcovnet.pytorch.onset_ar import targets


def teacher_forced_prev_patch_indices(
    target_patch_indices: torch.Tensor,
) -> torch.Tensor:
    """Return previous onset patch per decoder step (0 at the first step).

    Args:
        target_patch_indices: Ground-truth patch indices ``(B, T)``.

    Returns:
        Shifted int64 tensor ``(B, T)``.
    """
    shifted = torch.zeros_like(target_patch_indices, dtype=torch.long)
    if target_patch_indices.shape[1] > 1:
        shifted[:, 1:] = target_patch_indices[:, :-1].to(torch.long)
    return shifted


def apply_monotonic_pointer_mask(
    pointer_logits: torch.Tensor,
    prev_patch_indices: torch.Tensor,
) -> torch.Tensor:
    """Mask pointer logits strictly below ``prev_patch_indices``.

    Args:
        pointer_logits: Absolute pointer logits ``(B, T, P)``.
        prev_patch_indices: Previous patch indices ``(B, T)``.

    Returns:
        Masked float32 logits ``(B, T, P)``.
    """
    logits = pointer_logits.float()
    n_patches = int(logits.shape[-1])
    patch_ids = torch.arange(n_patches, device=logits.device, dtype=torch.long).view(
        1, 1, n_patches
    )
    prev = prev_patch_indices.to(torch.long).unsqueeze(-1)
    invalid = (patch_ids < prev).to(logits.dtype)
    return logits + invalid * (-1e9)


def apply_prev_relative_window(
    pointer_logits: torch.Tensor,
    prev_patch_indices: torch.Tensor,
    *,
    max_ahead: int,
) -> torch.Tensor:
    """Mask pointer logits strictly after ``prev + max_ahead`` when ``prev > 0``.

    Args:
        pointer_logits: Absolute pointer logits ``(B, T, P)``.
        prev_patch_indices: Previous patch indices ``(B, T)``.
        max_ahead: Inclusive forward window in patches (``<= 0`` is a no-op).

    Returns:
        Masked float32 logits ``(B, T, P)``.
    """
    if max_ahead <= 0:
        return pointer_logits.float()
    logits = pointer_logits.float()
    n_patches = int(logits.shape[-1])
    patch_ids = torch.arange(n_patches, device=logits.device, dtype=torch.long).view(
        1, 1, n_patches
    )
    prev = prev_patch_indices.to(torch.long).unsqueeze(-1)
    invalid = ((patch_ids > prev + int(max_ahead)) & (prev > 0)).to(logits.dtype)
    return logits + invalid * (-1e9)


def prev_relative_ce_step_mask(
    target_patch_indices: torch.Tensor,
    prev_patch_indices: torch.Tensor,
    *,
    max_ahead: int,
    onset_step_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero out onset steps whose target falls outside ``[prev, prev + max_ahead]``.

    Args:
        target_patch_indices: Target patch indices ``(B, T)``.
        prev_patch_indices: Previous patch indices ``(B, T)``.
        max_ahead: Forward window width (``<= 0`` returns ``onset_step_mask``).
        onset_step_mask: Float onset mask ``(B, T)``.

    Returns:
        Filtered float32 mask ``(B, T)``.
    """
    onset = onset_step_mask.float()
    if max_ahead <= 0:
        return onset
    target = target_patch_indices.to(torch.long)
    prev = prev_patch_indices.to(torch.long)
    in_window = (prev <= 0) | (target <= prev + int(max_ahead))
    return onset * in_window.to(torch.float32)


def apply_local_pointer_ce_mask(
    pointer_logits: torch.Tensor,
    target_patch_indices: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Mask pointer logits outside ``[target - radius, target + radius]``.

    Args:
        pointer_logits: Pointer logits ``(B, T, P)``.
        target_patch_indices: Target patch indices ``(B, T)``.
        radius: Inclusive half-width in patches (``<= 0`` is a no-op).

    Returns:
        Masked float32 logits ``(B, T, P)``.
    """
    if radius <= 0:
        return pointer_logits.float()
    logits = pointer_logits.float()
    n_patches = int(logits.shape[-1])
    patch_ids = torch.arange(n_patches, device=logits.device, dtype=torch.long).view(
        1, 1, n_patches
    )
    target = target_patch_indices.to(torch.long).unsqueeze(-1)
    invalid = (
        (patch_ids < target - int(radius)) | (patch_ids > target + int(radius))
    ).to(logits.dtype)
    return logits + invalid * (-1e9)


def apply_soft_distance_prior(
    pointer_logits: torch.Tensor,
    prev_patch_indices: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Subtract ``alpha * max(0, patch - prev)`` from logits when ``prev > 0``.

    Args:
        pointer_logits: Pointer logits ``(B, T, P)``.
        prev_patch_indices: Previous patch indices ``(B, T)``.
        alpha: Non-negative distance penalty slope.

    Returns:
        Adjusted float32 logits ``(B, T, P)``.
    """
    logits = pointer_logits.float()
    if float(alpha) <= 0.0:
        return logits
    n_patches = int(logits.shape[-1])
    patch_ids = torch.arange(n_patches, device=logits.device, dtype=torch.float32).view(
        1, 1, n_patches
    )
    prev = prev_patch_indices.float().unsqueeze(-1)
    ahead = nn_func.relu(patch_ids - prev)
    apply_mask = (prev > 0.0).to(logits.dtype)
    return logits - float(alpha) * ahead * apply_mask


def apply_gap_soft_distance_prior(
    gap_logits: torch.Tensor,
    prev_patch_indices: torch.Tensor,
    *,
    alpha: float,
    delta_lookup: torch.Tensor,
) -> torch.Tensor:
    """Subtract ``alpha * decode_delta(id)`` from gap logits when ``prev > 0``.

    Args:
        gap_logits: Gap logits ``(B, T, V)``.
        prev_patch_indices: Previous patch indices ``(B, T)``.
        alpha: Non-negative distance penalty slope.
        delta_lookup: Decoded patch deltas ``(V,)``.

    Returns:
        Adjusted float32 gap logits ``(B, T, V)``.
    """
    logits = gap_logits.float()
    if float(alpha) <= 0.0:
        return logits
    deltas = delta_lookup.to(device=logits.device, dtype=torch.float32).view(1, 1, -1)
    apply_mask = (prev_patch_indices.float().unsqueeze(-1) > 0.0).to(logits.dtype)
    return logits - float(alpha) * deltas * apply_mask


def build_token_class_weights_numpy(
    decoder_target_ids: np.ndarray,
    decoder_mask: np.ndarray,
    *,
    vocab_size: int,
    scheme: str,
    eos_token_weight_scale: float = 1.0,
) -> np.ndarray | None:
    """Compute optional per-vocab-id CE weights over active decoder targets.

    Args:
        decoder_target_ids: Target token id array.
        decoder_mask: Binary validity mask matching ``decoder_target_ids``.
        vocab_size: Vocabulary size.
        scheme: ``none``, ``inverse_freq``, or ``inverse_sqrt_freq``.
        eos_token_weight_scale: Multiplier applied to ``EOS_ID``.

    Returns:
        Float32 array of shape ``(vocab_size,)`` or ``None``.

    Raises:
        ValueError: If ``scheme`` is unsupported.
    """
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


def _soft_expected_patch_indices(pointer_logits: torch.Tensor) -> torch.Tensor:
    n_patches = int(pointer_logits.shape[-1])
    patch_indices = torch.arange(
        n_patches, device=pointer_logits.device, dtype=torch.float32
    )
    probs = torch.softmax(pointer_logits.float(), dim=-1)
    return torch.sum(probs * patch_indices, dim=-1)


def predicted_times_from_outputs(
    pointer_logits: torch.Tensor,
    residual_sec: torch.Tensor,
    *,
    patch_frames: int,
    hop_sec: float,
    use_soft_expected: bool = False,
    use_ste: bool = False,
) -> torch.Tensor:
    """Convert pointer logits and residual seconds to predicted onset times.

    Args:
        pointer_logits: Pointer logits ``(B, T, P)``.
        residual_sec: Within-patch residual seconds ``(B, T)``.
        patch_frames: Frames per patch.
        hop_sec: Seconds per hop frame.
        use_soft_expected: Use softmax expectation instead of argmax.
        use_ste: Hard argmax forward with soft expectation backward (STE).

    Returns:
        Predicted onset times ``(B, T)`` in seconds.
    """
    logits = pointer_logits.float()
    res = residual_sec.float()
    patch_duration = float(patch_frames) * float(hop_sec)
    hard_patch = torch.argmax(logits, dim=-1).float()
    if use_ste:
        soft_patch = _soft_expected_patch_indices(logits)
        patch_idx = soft_patch + (hard_patch - soft_patch).detach()
    elif use_soft_expected:
        patch_idx = _soft_expected_patch_indices(logits)
    else:
        patch_idx = hard_patch
    return patch_idx * patch_duration + res


def compute_ar_onset_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    patch_frames: int,
    hop_sec: float,
    lambda_time: float = 0.0,
    lambda_residual: float = 0.0,
    pointer_loss_weight: float = 1.0,
    length_normalize_ce: bool = True,
    token_class_weights: torch.Tensor | None = None,
    use_soft_pointer_time: bool = False,
    use_ste_pointer_time: bool = False,
    time_loss_correct_patch_only: bool = False,
    pointer_local_ce_radius: int = 0,
    pointer_local_ce_anchor: str = "target",
    pointer_soft_distance_alpha: float = 0.0,
    monotonic_pointer: bool = False,
    gap_alignment: bool = False,
    gap_loss_weight: float = 1.0,
    gap_delta_lookup: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute teacher-forced AR onset loss and diagnostic scalar components.

    Args:
        outputs: Model head dictionary.
        batch: Teacher-forced batch tensor dictionary.
        patch_frames: Frames per encoder patch.
        hop_sec: Seconds per hop frame.
        lambda_time: Weight on absolute onset time L1 loss.
        lambda_residual: Weight on within-patch residual MSE loss.
        pointer_loss_weight: Weight on absolute pointer CE loss.
        length_normalize_ce: Divide token CE by valid token count when True.
        token_class_weights: Optional per-token CE weights ``(V,)``.
        use_soft_pointer_time: Soft expected patch for ``lambda_time``.
        use_ste_pointer_time: Straight-through estimator for ``lambda_time``.
        time_loss_correct_patch_only: Mask time L1 to steps with exact patch match.
        pointer_local_ce_radius: Local pointer CE window radius (``0`` = disabled).
        pointer_local_ce_anchor: ``target`` or ``prev``.
        pointer_soft_distance_alpha: Soft distance penalty slope.
        monotonic_pointer: Apply monotonic mask to absolute pointer logits.
        gap_alignment: Use relative gap CE as primary alignment loss.
        gap_loss_weight: Weight on gap CE when ``gap_alignment`` is True.
        gap_delta_lookup: Decoded delta lookup table for gap alignment.

    Returns:
        Tuple ``(total_loss, components_dict)``.

    Raises:
        ValueError: If required heads or lookup tables are missing.
    """
    token_logits = outputs["token_logits"].float()
    residual_sec = outputs["residual_sec"].float()
    pointer_logits = (
        outputs["pointer_logits"].float() if "pointer_logits" in outputs else None
    )
    gap_logits = outputs["gap_logits"].float() if "gap_logits" in outputs else None

    decoder_target_ids = batch["decoder_target_ids"].to(torch.long)
    decoder_mask = batch["decoder_mask"].float()
    onset_step_mask = batch["onset_step_mask"].float()
    target_patch_indices = batch["target_patch_indices"].to(torch.long)
    target_times = batch["target_times"].float()
    target_residual_sec = batch["target_residual_sec"].float()

    prev_patches = teacher_forced_prev_patch_indices(target_patch_indices)
    if pointer_logits is not None and not gap_alignment:
        if monotonic_pointer:
            pointer_logits = apply_monotonic_pointer_mask(pointer_logits, prev_patches)
        pointer_logits = apply_soft_distance_prior(
            pointer_logits,
            prev_patches,
            alpha=pointer_soft_distance_alpha,
        )
    elif pointer_logits is not None and monotonic_pointer:
        pointer_logits = apply_monotonic_pointer_mask(pointer_logits, prev_patches)

    if gap_alignment and gap_logits is not None:
        if gap_delta_lookup is None:
            raise ValueError("gap_alignment requires gap_delta_lookup")
        gap_logits = apply_gap_soft_distance_prior(
            gap_logits,
            prev_patches,
            alpha=pointer_soft_distance_alpha,
            delta_lookup=gap_delta_lookup,
        )

    token_ce = nn_func.cross_entropy(
        token_logits.reshape(-1, token_logits.shape[-1]),
        decoder_target_ids.reshape(-1),
        reduction="none",
    ).reshape(decoder_target_ids.shape)
    if token_class_weights is not None:
        w = token_class_weights.to(device=token_ce.device, dtype=torch.float32)
        token_ce = token_ce * w[decoder_target_ids]
    token_ce = token_ce * decoder_mask
    token_count = decoder_mask.sum() + 1e-9
    token_loss = token_ce.sum() / token_count if length_normalize_ce else token_ce.sum()

    onset_count = onset_step_mask.sum() + 1e-9
    zero = torch.zeros((), device=token_logits.device, dtype=torch.float32)
    gap_loss = zero
    pointer_loss = zero
    alignment_term = zero

    if gap_alignment:
        if gap_logits is None:
            raise ValueError("gap_alignment requires outputs['gap_logits']")
        if gap_delta_lookup is None:
            raise ValueError("gap_alignment requires gap_delta_lookup")
        target_gap_ids = batch["target_gap_ids"].to(torch.long)
        gap_ce = nn_func.cross_entropy(
            gap_logits.reshape(-1, gap_logits.shape[-1]),
            target_gap_ids.reshape(-1),
            reduction="none",
        ).reshape(target_gap_ids.shape)
        gap_loss = (gap_ce * onset_step_mask).sum() / onset_count
        alignment_term = float(gap_loss_weight) * gap_loss
        pointer_loss = gap_loss
        if pointer_logits is not None and float(pointer_loss_weight) > 0.0:
            abs_ce = nn_func.cross_entropy(
                pointer_logits.reshape(-1, pointer_logits.shape[-1]),
                target_patch_indices.reshape(-1),
                reduction="none",
            ).reshape(target_patch_indices.shape)
            abs_ptr_loss = (abs_ce * onset_step_mask).sum() / onset_count
            alignment_term = alignment_term + float(pointer_loss_weight) * abs_ptr_loss
        n_patches = int(batch["patch_mask"].shape[1])
        gap_ids = torch.argmax(gap_logits, dim=-1)
        deltas = gap_delta_lookup.to(gap_ids.device)[gap_ids]
        hard_patch = (prev_patches + deltas).clamp(0, max(0, n_patches - 1))
        pred_times = (
            hard_patch.float() * (float(patch_frames) * float(hop_sec)) + residual_sec
        )
    else:
        if pointer_logits is None:
            raise ValueError("pointer alignment requires outputs['pointer_logits']")
        anchor = str(pointer_local_ce_anchor or "target").lower()
        radius = int(pointer_local_ce_radius)
        pointer_ce_mask = onset_step_mask
        if radius > 0 and anchor == "prev":
            pointer_logits = apply_prev_relative_window(
                pointer_logits, prev_patches, max_ahead=radius
            )
            pointer_logits_for_ce = pointer_logits
            pointer_ce_mask = prev_relative_ce_step_mask(
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
        ptr_ce = nn_func.cross_entropy(
            pointer_logits_for_ce.reshape(-1, pointer_logits_for_ce.shape[-1]),
            target_patch_indices.reshape(-1),
            reduction="none",
        ).reshape(target_patch_indices.shape)
        pointer_loss = (ptr_ce * pointer_ce_mask).sum() / (pointer_ce_mask.sum() + 1e-9)
        alignment_term = float(pointer_loss_weight) * pointer_loss
        pred_times = predicted_times_from_outputs(
            pointer_logits,
            residual_sec,
            patch_frames=patch_frames,
            hop_sec=hop_sec,
            use_soft_expected=use_soft_pointer_time and not use_ste_pointer_time,
            use_ste=use_ste_pointer_time,
        )
        hard_patch = torch.argmax(pointer_logits, dim=-1)

    time_mask = onset_step_mask
    if time_loss_correct_patch_only:
        time_mask = onset_step_mask * (hard_patch == target_patch_indices).float()
    time_loss = (torch.abs(pred_times - target_times) * time_mask).sum() / (
        time_mask.sum() + 1e-9
    )
    residual_loss = (
        torch.square(residual_sec - target_residual_sec) * onset_step_mask
    ).sum() / onset_count

    total_loss = (
        token_loss
        + alignment_term
        + float(lambda_time) * time_loss
        + float(lambda_residual) * residual_loss
    )
    return total_loss, {
        "token_loss": token_loss,
        "pointer_loss": pointer_loss,
        "gap_loss": gap_loss,
        "time_loss": time_loss,
        "residual_loss": residual_loss,
    }


def masked_token_accuracy(
    token_logits: torch.Tensor,
    decoder_target_ids: torch.Tensor,
    decoder_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute top-1 token accuracy over non-padded decoder positions.

    Args:
        token_logits: Token logits ``(B, T, V)``.
        decoder_target_ids: Target token ids ``(B, T)``.
        decoder_mask: Binary validity mask ``(B, T)``.

    Returns:
        Scalar float32 accuracy tensor.
    """
    preds = torch.argmax(token_logits, dim=-1)
    correct = (
        preds == decoder_target_ids.to(torch.long)
    ).float() * decoder_mask.float()
    return correct.sum() / (decoder_mask.float().sum() + 1e-9)


def masked_alignment_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    onset_step_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute top-1 pointer or gap accuracy over onset decoder positions.

    Args:
        logits: Pointer or gap logits ``(B, T, C)``.
        target_ids: Target class indices ``(B, T)``.
        onset_step_mask: Binary onset step mask ``(B, T)``.

    Returns:
        Scalar float32 accuracy tensor.
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == target_ids.to(torch.long)).float() * onset_step_mask.float()
    return correct.sum() / (onset_step_mask.float().sum() + 1e-9)
