"""Autoregressive KV-cached decoding and event evaluation for PyTorch AR onset."""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np
import torch

from stepcovnet import timing_match as timing_match_lib
from stepcovnet.ddc import peak_pick
from stepcovnet.pytorch.onset_ar import config, datasets, models, targets

ArTimeSource = Literal["pointer_residual", "tokens"]


@dataclasses.dataclass(frozen=True)
class ArDecodeStats:
    """Free-running autoregressive decode summary.

    Attributes:
        times: Predicted onset times in seconds.
        n_forward_steps: Number of incremental decoder steps executed.
        n_onset_tokens: Onset tokens emitted before stopping.
        stopped_on_eos: Whether decode stopped on ``<EOS>``.
        onset_token_ids: Emitted onset token ids.
        eos_prob_trace: Per-step ``<EOS>`` probability before length control.
    """

    times: np.ndarray
    n_forward_steps: int
    n_onset_tokens: int
    stopped_on_eos: bool
    onset_token_ids: np.ndarray
    eos_prob_trace: np.ndarray


@dataclasses.dataclass(frozen=True)
class ArLengthControl:
    """Decode-time constraints on free-run sequence length.

    Attributes:
        eos_logit_bias: Additive bias applied to ``<EOS>`` logit.
        min_onset_tokens: Minimum emitted onset tokens before ``<EOS>`` is allowed.
    """

    eos_logit_bias: float = 0.0
    min_onset_tokens: int = 0

    def is_active(self) -> bool:
        """Return whether this length control modifies greedy decoding.

        Returns:
            True when either bias or minimum token count is active.
        """
        return self.eos_logit_bias != 0.0 or self.min_onset_tokens > 0


@dataclasses.dataclass(frozen=True)
class ArEvalReport:
    """Aggregated free-run AR onset evaluation metrics.

    Attributes:
        event_f1: Micro ±tolerance event F1 (Hungarian/greedy 1:1 matching).
        timing_match: Micro ordered timing match rate.
        n_samples: Number of evaluated charts.
        total_pred: Total predicted onsets across charts.
        total_gt: Total ground-truth onsets across charts.
        eos_stop_rate: Fraction of charts that terminated on ``<EOS>``.
    """

    event_f1: float
    timing_match: float
    n_samples: int
    total_pred: int
    total_gt: int
    eos_stop_rate: float

    def as_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-compatible dictionary.

        Returns:
            Plain dict of metrics.
        """
        return dataclasses.asdict(self)


def eos_probability(token_logits: np.ndarray, *, eos_id: int = targets.EOS_ID) -> float:
    """Return softmax probability of ``<EOS>`` for one step's token logits.

    Args:
        token_logits: Unnormalized token scores ``(V,)``.
        eos_id: Vocabulary id of ``<EOS>``.

    Returns:
        Probability in ``[0, 1]``.
    """
    logits = np.asarray(token_logits, dtype=np.float64).reshape(-1)
    shifted = np.exp(logits - float(np.max(logits)))
    return float(shifted[eos_id] / max(float(np.sum(shifted)), 1e-12))


def select_next_token(
    token_logits: np.ndarray,
    *,
    eos_id: int = targets.EOS_ID,
    n_emitted: int = 0,
    length_control: ArLengthControl | None = None,
) -> int:
    """Return greedy next token id after applying optional length control.

    Args:
        token_logits: Single-step logits ``(V,)``.
        eos_id: End-of-sequence token id.
        n_emitted: Number of onset tokens emitted so far.
        length_control: Optional decode-time ``<EOS>`` bias / minimum count.

    Returns:
        Selected integer token id.
    """
    logits = np.asarray(token_logits, dtype=np.float32)
    if length_control is None or not length_control.is_active():
        return int(np.argmax(logits))
    adjusted = logits.copy()
    if n_emitted < length_control.min_onset_tokens:
        adjusted[eos_id] = -np.inf
    else:
        adjusted[eos_id] += float(length_control.eos_logit_bias)
    return int(np.argmax(adjusted))


def _argmax_pointer_patch(
    pointer_logits: np.ndarray,
    *,
    prev_patch: int,
    monotonic: bool,
    max_ahead: int = 0,
    soft_distance_alpha: float = 0.0,
) -> int:
    logits = np.asarray(pointer_logits, dtype=np.float32).copy()
    prev = max(0, int(prev_patch))
    if monotonic and prev > 0:
        logits[:prev] = -1e9
    if float(soft_distance_alpha) > 0.0 and prev > 0:
        ahead = np.maximum(
            np.arange(logits.shape[-1], dtype=np.float32) - float(prev), 0.0
        )
        logits = logits - float(soft_distance_alpha) * ahead
    if max_ahead > 0 and prev > 0:
        hi = prev + int(max_ahead)
        if hi + 1 < logits.shape[-1]:
            logits[hi + 1 :] = -1e9
    return int(np.argmax(logits))


def _argmax_gap_patch(
    gap_logits: np.ndarray,
    *,
    prev_patch: int,
    gap_vocab: targets.PatchGapVocab,
    max_patch: int,
    soft_distance_alpha: float = 0.0,
) -> int:
    logits = np.asarray(gap_logits, dtype=np.float32).copy()
    if float(soft_distance_alpha) > 0.0 and int(prev_patch) > 0:
        deltas = targets.gap_delta_lookup_table(gap_vocab).astype(np.float32)
        logits = logits - float(soft_distance_alpha) * deltas
    gap_id = int(np.argmax(logits))
    delta = gap_vocab.decode_delta(gap_id)
    return min(max(int(prev_patch) + delta, 0), int(max_patch))


@torch.no_grad()
def decode_autoregressive_kv(
    model: models.ArOnsetModel,
    mert_patches: np.ndarray,
    patch_mask: np.ndarray,
    *,
    experiment_config: config.ArExperimentConfig | None = None,
    max_decoder_len: int | None = None,
    time_source: ArTimeSource = "pointer_residual",
    length_control: ArLengthControl | None = None,
    density_scalar: float | None = None,
    soft_distance_alpha: float | None = None,
    device: torch.device | None = None,
) -> ArDecodeStats:
    """Free-run autoregressive decode using true O(L) per-layer KV caching.

    Args:
        model: Trained :class:`models.ArOnsetModel`.
        mert_patches: Patched MERT features ``(P, patch_dim)`` or ``(1, P, patch_dim)``.
        patch_mask: Patch validity mask ``(P,)`` or ``(1, P)``.
        experiment_config: Optional config override (defaults to ``model.experiment_config``).
        max_decoder_len: Optional hard cap on decoder steps.
        time_source: ``pointer_residual`` or ``tokens``.
        length_control: Optional ``<EOS>`` length constraints.
        density_scalar: Optional density conditioning scalar.
        soft_distance_alpha: Optional override for soft distance penalty.
        device: Optional device (defaults to model's parameter device).

    Returns:
        :class:`ArDecodeStats` containing predicted times and trace diagnostics.
    """
    cfg = experiment_config or model.experiment_config
    target_device = device or next(model.parameters()).device
    model.eval()

    mp = np.asarray(mert_patches, dtype=np.float32)
    pm = np.asarray(patch_mask, dtype=np.float32)
    if mp.ndim == 2:
        mp = mp[np.newaxis, ...]
    if pm.ndim == 1:
        pm = pm[np.newaxis, ...]

    cap = int(max_decoder_len or cfg.max_decoder_len())
    mp_t = torch.from_numpy(mp).to(target_device)
    pm_t = torch.from_numpy(pm).to(target_device)
    density_t: torch.Tensor | None = None
    if config.density_conditioning_active(cfg.model):
        val = 0.0 if density_scalar is None else float(density_scalar)
        density_t = torch.tensor([[val]], dtype=torch.float32, device=target_device)

    cache = model.init_kv_cache(
        mp_t,
        pm_t,
        max_decoder_len=cap,
        density_scalar=density_t,
    )
    patch_duration = float(cfg.model.patch_frames) * float(cfg.dataset.hop_sec)
    gap_alignment = config.gap_alignment_active(cfg.model)
    gap_vocab = cfg.build_gap_vocab() if gap_alignment else None
    max_patch = max(int(np.sum(pm[0] > 0.5)) - 1, 0)
    monotonic = bool(cfg.model.monotonic_pointer)
    max_ahead = 0 if gap_alignment else config.pointer_decode_max_ahead(cfg.run)
    alpha = (
        float(cfg.run.pointer_soft_distance_alpha)
        if soft_distance_alpha is None
        else float(soft_distance_alpha)
    )

    pointer_times: list[float] = []
    onset_token_ids: list[int] = []
    eos_probs: list[float] = []
    prev_patch = 0
    n_forward_steps = 0
    stopped_on_eos = False
    cur_token = targets.BOS_ID

    for pos in range(cap - 1):
        n_forward_steps += 1
        tok_t = torch.tensor([[cur_token]], dtype=torch.long, device=target_device)
        prev_t = (
            torch.tensor([[prev_patch]], dtype=torch.long, device=target_device)
            if model.content_gap
            else None
        )
        step_out = model.decode_step_cached(
            tok_t,
            pos,
            cache,
            prev_patch_indices=prev_t,
        )
        token_logits = step_out["token_logits"][0, 0].detach().cpu().numpy()
        residual_sec = float(step_out["residual_sec"][0, 0].detach().cpu().item())
        eos_probs.append(eos_probability(token_logits, eos_id=targets.EOS_ID))
        next_token = select_next_token(
            token_logits,
            eos_id=targets.EOS_ID,
            n_emitted=len(onset_token_ids),
            length_control=length_control,
        )
        if next_token == targets.EOS_ID:
            stopped_on_eos = True
            break
        onset_token_ids.append(next_token)
        if gap_alignment:
            assert gap_vocab is not None
            gap_logits = step_out["gap_logits"][0, 0].detach().cpu().numpy()
            patch_idx = _argmax_gap_patch(
                gap_logits,
                prev_patch=prev_patch,
                gap_vocab=gap_vocab,
                max_patch=max_patch,
                soft_distance_alpha=alpha,
            )
            prev_patch = patch_idx
        else:
            pointer_logits = step_out["pointer_logits"][0, 0].detach().cpu().numpy()
            patch_idx = _argmax_pointer_patch(
                pointer_logits,
                prev_patch=prev_patch,
                monotonic=monotonic,
                max_ahead=max_ahead,
                soft_distance_alpha=alpha,
            )
            if monotonic:
                prev_patch = patch_idx
        pointer_times.append(float(patch_idx) * patch_duration + residual_sec)
        cur_token = next_token

    token_arr = np.asarray(onset_token_ids, dtype=np.int32)
    if time_source == "tokens":
        max_frame = max(1, int(np.sum(pm[0] > 0.5)) * int(cfg.model.patch_frames))
        times = targets.decode_token_sequence_to_times(
            token_arr,
            hop_sec=cfg.dataset.hop_sec,
            vocab=cfg.build_vocab(),
            max_frame=max_frame,
        )
    else:
        times = np.asarray(pointer_times, dtype=np.float32)

    return ArDecodeStats(
        times=times,
        n_forward_steps=n_forward_steps,
        n_onset_tokens=len(onset_token_ids),
        stopped_on_eos=stopped_on_eos,
        onset_token_ids=token_arr,
        eos_prob_trace=np.asarray(eos_probs, dtype=np.float32),
    )


def evaluate_ar_samples(
    model: models.ArOnsetModel,
    samples: list[datasets.ArSample],
    *,
    tolerance_sec: float | None = None,
    max_decoder_len: int | None = None,
    device: torch.device | None = None,
) -> ArEvalReport:
    """Evaluate free-run KV-cached AR decode on a list of samples.

    Args:
        model: Trained AR onset model.
        samples: Evaluation samples.
        tolerance_sec: Onset matching tolerance in seconds.
        max_decoder_len: Optional decode step cap.
        device: Optional compute device.

    Returns:
        Aggregated :class:`ArEvalReport`.
    """
    cfg = model.experiment_config
    tol = float(cfg.run.tolerance_sec if tolerance_sec is None else tolerance_sec)
    pooled = peak_pick.OnsetMatchCounts(0, 0, 0)
    tm_matched = 0
    tm_ref = 0
    tm_pred = 0
    eos_stops = 0

    for sample in samples:
        patch_mask = np.ones((sample.n_patches,), dtype=np.float32)
        stats = decode_autoregressive_kv(
            model,
            sample.mert_patches,
            patch_mask,
            max_decoder_len=max_decoder_len,
            density_scalar=sample.density_scalar,
            device=device,
        )
        if stats.stopped_on_eos:
            eos_stops += 1
        pred_times = np.sort(np.asarray(stats.times, dtype=np.float64))
        gt_times = np.asarray(sample.gt_times_sec, dtype=np.float64)
        pooled = peak_pick.add_counts(
            pooled,
            peak_pick.match_onsets(pred_times, gt_times, tolerance_sec=tol),
        )
        m_i, ref_i = timing_match_lib.timing_match_counts_numpy(
            pred_times, gt_times, tolerance_sec=tol
        )
        tm_matched += m_i
        tm_ref += ref_i
        tm_pred += int(pred_times.size)

    n = max(1, len(samples))
    return ArEvalReport(
        event_f1=peak_pick.f_score_m(pooled),
        timing_match=timing_match_lib.micro_timing_match_rate(
            tm_matched, tm_ref, tm_pred
        ),
        n_samples=len(samples),
        total_pred=tm_pred,
        total_gt=tm_ref,
        eos_stop_rate=float(eos_stops) / float(n),
    )
