"""Snap DDC Hamming peaks onto the DDCL/ITGPT ``M-slot48`` grid.

This is a POST conversion of accepted peak times, not DDC trained with
48-slot BCE. Ground-truth slots use the same ``times_to_slot_matrix`` path as
DDCL so Dataset A comparisons share one label grid. Onset times must stay
float64; float32 quantization moves 48-slot labels and the beat-shuffle null.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddc import datasets
from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import evaluation as ddcl_eval
from stepcovnet.ddcl import slots

CONVERSION = "ddc_hamming_peak_snap"


def _slot_matrix_from_times(
    times_sec: np.ndarray,
    offset_sec: float,
    segments: Sequence[prep_models.BpmSegment],
) -> np.ndarray:
    """Return ``(n_beats, 48)`` occupancy, or an empty matrix when no times.

    Args:
        times_sec: Onset or peak times in seconds.
        offset_sec: Simfile ``#OFFSET``.
        segments: BPM ladder.

    Returns:
        Float32 0/1 matrix. Shape ``(0, 48)`` when ``times_sec`` is empty.
    """
    times = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return np.zeros((0, ddcl_constants.N_SLOTS), dtype=np.float32)
    return slots.times_to_slot_matrix(times, offset_sec, segments)


def align_slot_matrices(
    pred: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Align pred occupancy to the GT beat span.

    Pred rows past the last GT beat are not padded onto the label grid
    (that would change ``n_beats`` and the beat-shuffle null). Their
    occupied slots are returned as extra false positives.

    Args:
        pred: Predicted occupancy ``(n_pred_beats, 48)``.
        target: Ground-truth occupancy ``(n_tgt_beats, 48)``.

    Returns:
        ``(pred, target, extra_false_positives)`` where pred/target share
        the GT beat length.

    Raises:
        ValueError: If either matrix is not rank-2 with 48 slots, or GT
            has no beats.
    """
    pred_arr = np.asarray(pred, dtype=np.float32)
    tgt_arr = np.asarray(target, dtype=np.float32)
    if pred_arr.ndim != 2 or pred_arr.shape[1] != ddcl_constants.N_SLOTS:
        raise ValueError(f"pred must be (n_beats, 48), got {pred_arr.shape}")
    if tgt_arr.ndim != 2 or tgt_arr.shape[1] != ddcl_constants.N_SLOTS:
        raise ValueError(f"target must be (n_beats, 48), got {tgt_arr.shape}")
    n_beats = int(tgt_arr.shape[0])
    if n_beats < 1:
        raise ValueError("target must contain at least one beat")
    extra = pred_arr[n_beats:]
    extra_fp = int((extra >= 0.5).sum()) if extra.size else 0
    pred_out = np.zeros((n_beats, ddcl_constants.N_SLOTS), dtype=np.float32)
    copy_n = min(int(pred_arr.shape[0]), n_beats)
    pred_out[:copy_n] = pred_arr[:copy_n]
    return pred_out, tgt_arr, extra_fp


def evaluate_peak_times_as_slot48(
    charts: list[datasets.PlacementChart],
    pred_times: Sequence[np.ndarray],
    *,
    seed: int = 42,
) -> ddcl_eval.Slot48EvalReport:
    """Pool ``M-slot48`` by snapping peak times onto the beat grid.

    Args:
        charts: DDC charts with ``gt_times``, ``offset_sec``, and BPM segments.
        pred_times: Hamming peak times aligned with ``charts``.
        seed: Beat-shuffle null seed.

    Returns:
        Pooled ``M-slot48`` report (binary peaks; max-F1 equals F1@0.5).

    Raises:
        ValueError: If ``charts`` is empty, lengths differ, or BPM is missing.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    if len(pred_times) != len(charts):
        raise ValueError("pred_times must align with charts")
    rng = np.random.default_rng(seed)
    pred_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    extra_fp = 0
    for chart, times in zip(charts, pred_times, strict=True):
        if not chart.bpm_segments:
            raise ValueError(f"missing BPM segments for {chart.song_key}")
        target = _slot_matrix_from_times(
            np.asarray(chart.gt_times, dtype=np.float64),
            chart.offset_sec,
            chart.bpm_segments,
        )
        if target.shape[0] < 1:
            raise ValueError(f"empty slot grid for {chart.song_key}")
        pred = _slot_matrix_from_times(times, chart.offset_sec, chart.bpm_segments)
        pred_aligned, tgt_aligned, extra = align_slot_matrices(pred, target)
        extra_fp += extra
        pred_parts.append(pred_aligned)
        tgt_parts.append(tgt_aligned)
    pred_all = np.concatenate(pred_parts, axis=0)
    tgt_all = np.concatenate(tgt_parts, axis=0)
    counts_05 = ddcl_eval.counts_at_threshold(
        pred_all, tgt_all, ddcl_constants.THRESHOLD_05
    )
    if extra_fp:
        counts_05 = ddcl_eval.Slot48Counts(
            true_positives=counts_05.true_positives,
            false_positives=counts_05.false_positives + extra_fp,
            false_negatives=counts_05.false_negatives,
        )
    null_pred = ddcl_eval.shuffle_slot_null(tgt_all, rng)
    null_counts = ddcl_eval.counts_at_threshold(
        null_pred, tgt_all, ddcl_constants.THRESHOLD_05
    )
    return ddcl_eval.Slot48EvalReport(
        f1_at_05=counts_05.f_score,
        f1_max=counts_05.f_score,
        best_threshold=ddcl_constants.THRESHOLD_05,
        null_f1_at_05=null_counts.f_score,
        n_charts=len(charts),
        n_beats=int(tgt_all.shape[0]),
        counts_at_05=counts_05,
    )


def report_as_dict(report: ddcl_eval.Slot48EvalReport) -> dict:
    """Serialize a peak-snap ``M-slot48`` report.

    Args:
        report: Pooled scores.

    Returns:
        JSON-serializable mapping without DDCL Table 2 published numbers.
    """
    return {
        "metric": "M-slot48",
        "conversion": CONVERSION,
        "f1_at_05": report.f1_at_05,
        "f1_max": report.f1_max,
        "best_threshold": report.best_threshold,
        "null_f1_at_05": report.null_f1_at_05,
        "skill_f1_at_05": report.f1_at_05 - report.null_f1_at_05,
        "n_charts": report.n_charts,
        "n_beats": report.n_beats,
        "true_positives": report.counts_at_05.true_positives,
        "false_positives": report.counts_at_05.false_positives,
        "false_negatives": report.counts_at_05.false_negatives,
        "citation": "donahue2017ddc",
        "note": (
            "Hamming peak times snapped onto 48 slots/beat; DDC was not "
            "trained with slot BCE"
        ),
    }
