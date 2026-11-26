"""Zero-TF ``M-slot48`` evaluation for PyTorch DDCL placement."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stepcovnet.ddcl import constants, datasets
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.itgpt import evaluation as slot_eval


@torch.no_grad()
def predict_chart_slots(
    model: nn.Module,
    chart: datasets.DdclChart,
    *,
    batch_size: int = constants.PREDICT_BEAT_BATCH,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Predict ``(n_beats, 48)`` slot probabilities for one chart in beat chunks.

    Args:
        model: PyTorch DDCL placement model.
        chart: Loaded beat-grid chart.
        batch_size: Beats per forward chunk (avoids VRAM OOM on long songs).
        device: Optional inference device.
        amp_dtype: Optional autocast dtype (default fp32).

    Returns:
        Float32 array of shape ``(n_beats, 48)``.
    """
    if device is None:
        first_param = next(model.parameters(), None)
        target_device = (
            first_param.device if first_param is not None else torch.device("cpu")
        )
    else:
        target_device = device
    was_training = model.training
    model.eval()
    try:
        n_beats = chart.n_beats
        chunks: list[np.ndarray] = []
        step = max(1, int(batch_size))
        for start in range(0, n_beats, step):
            indices = np.arange(start, min(n_beats, start + step), dtype=np.int64)
            inputs_np = datasets.windows_for_indices(chart, indices)
            inputs = {
                k: torch.as_tensor(v, dtype=torch.float32, device=target_device)
                for k, v in inputs_np.items()
            }
            with runtime.autocast_context(target_device, amp_dtype):
                probs = model(inputs)
            chunks.append(probs.detach().float().cpu().numpy())
        return np.concatenate(chunks, axis=0)
    finally:
        if was_training:
            model.train()


def evaluate_slot48(
    model: nn.Module,
    charts: list[datasets.DdclChart],
    *,
    seed: int = 42,
    thresholds: tuple[float, ...] = slot_eval.DEFAULT_THRESHOLDS,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> slot_eval.Slot48EvalReport:
    """Predict every val chart and pool ``M-slot48`` F1.

    Args:
        model: Trained PyTorch DDCL model.
        charts: Loaded validation charts.
        seed: Null-shuffle seed.
        thresholds: Candidate threshold sweep for ``f1_max``.
        device: Optional inference device.
        amp_dtype: Optional autocast dtype.

    Returns:
        Pooled :class:`slot_eval.Slot48EvalReport`.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    rng = np.random.default_rng(seed)
    pred_parts = [
        predict_chart_slots(model, chart, device=device, amp_dtype=amp_dtype)
        for chart in charts
    ]
    tgt_parts = [chart.slots for chart in charts]
    pred_all = np.concatenate(pred_parts, axis=0)
    tgt_all = np.concatenate(tgt_parts, axis=0)
    counts_05 = slot_eval.counts_at_threshold(pred_all, tgt_all, constants.THRESHOLD_05)
    best_threshold = constants.THRESHOLD_05
    best_f1 = counts_05.f_score
    for threshold in thresholds:
        score = slot_eval.counts_at_threshold(pred_all, tgt_all, threshold).f_score
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    null_pred = slot_eval.shuffle_slot_null(tgt_all, rng)
    null_counts = slot_eval.counts_at_threshold(
        null_pred, tgt_all, constants.THRESHOLD_05
    )
    return slot_eval.Slot48EvalReport(
        f1_at_05=counts_05.f_score,
        f1_max=best_f1,
        best_threshold=best_threshold,
        null_f1_at_05=null_counts.f_score,
        n_charts=len(charts),
        n_beats=int(tgt_all.shape[0]),
        counts_at_05=counts_05,
    )


def report_as_dict(
    report: slot_eval.Slot48EvalReport, *, weights: str
) -> dict[str, object]:
    """Serialize a DDCL report with Table 2 citations.

    Args:
        report: Pooled ``M-slot48`` scores.
        weights: Checkpoint tag.

    Returns:
        JSON-serializable mapping.
    """
    payload = report.as_dict()
    payload["citation"] = "omalley2025ddcl"
    payload["published_f1_at_05_expanded_fraxtil"] = (
        constants.PUBLISHED_F1_AT_05_EXPANDED
    )
    payload["published_f1_max_expanded_fraxtil"] = constants.PUBLISHED_F1_MAX_EXPANDED
    payload["weights"] = weights
    payload["framework"] = "pytorch"
    return payload
