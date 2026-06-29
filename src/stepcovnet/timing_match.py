"""Unified onset timing quality: ordered match @ tolerance (primary scoreboard metric).

Every formulation (AR, dense peak-pick, event slots) exports a sorted list of
predicted onset times in seconds. Quality is how many reference onsets match in
order within ``tolerance_sec`` (default 20 ms). The rate is
``n_matched / max(n_pred, n_ref)`` so extra or missing predictions reduce the
score even when aligned pairs are perfect.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TOLERANCE_SEC = 0.02
TIMING_MATCH_METRIC_NAME = "timing_match"


def reference_times_from_mask(
    ref_times: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray:
    """Extract sorted reference onset times from a padded batch row."""
    times = np.asarray(ref_times, dtype=np.float64).reshape(-1)
    mask = np.asarray(ref_mask, dtype=np.float64).reshape(-1) > 0.5
    kept = times[mask]
    if kept.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sort(kept).astype(np.float32)


def timing_match_counts_numpy(
    pred_times: np.ndarray,
    ref_times: np.ndarray,
    *,
    tolerance_sec: float,
) -> tuple[int, int]:
    """Count ordered matches ``|pred[i] - ref[i]| <= tolerance``.

    Returns ``(n_matched, n_ref)``. Only ``min(n_pred, n_ref)`` pairs are
    compared; use :func:`timing_match_rate_from_counts` for the penalized rate.
    """
    pred = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_times, dtype=np.float64).reshape(-1)
    n_ref = int(ref.size)
    if n_ref == 0:
        return 0, 0
    n_compare = min(int(pred.size), n_ref)
    if n_compare == 0:
        return 0, n_ref
    diffs = np.abs(pred[:n_compare] - ref[:n_compare])
    n_matched = int(np.sum(diffs <= tolerance_sec))
    return n_matched, n_ref


def timing_match_denom(n_pred: int, n_ref: int) -> int:
    """Denominator for timing match rate (penalizes count mismatch)."""
    return max(int(n_pred), int(n_ref))


def timing_match_rate_from_counts(
    n_matched: int,
    n_pred: int,
    n_ref: int,
) -> float:
    denom = timing_match_denom(n_pred, n_ref)
    if denom == 0:
        return 0.0
    return float(n_matched) / float(denom)


def timing_match_rate_numpy(
    pred_times: np.ndarray,
    ref_times: np.ndarray,
    *,
    tolerance_sec: float,
) -> float:
    pred = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    n_matched, n_ref = timing_match_counts_numpy(
        pred_times,
        ref_times,
        tolerance_sec=tolerance_sec,
    )
    return timing_match_rate_from_counts(
        n_matched,
        int(pred.size),
        n_ref,
    )


def timing_match_report(
    pred_times: np.ndarray,
    ref_times: np.ndarray,
    *,
    tolerance_sec: float,
) -> dict[str, float | int]:
    pred = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    n_matched, n_ref = timing_match_counts_numpy(
        pred_times,
        ref_times,
        tolerance_sec=tolerance_sec,
    )
    n_pred = int(pred.size)
    n_denom = timing_match_denom(n_pred, n_ref)
    rate = timing_match_rate_from_counts(n_matched, n_pred, n_ref)
    return {
        "n_matched": n_matched,
        "n_pred": n_pred,
        "n_ref": n_ref,
        "n_denom": n_denom,
        "rate": rate,
        "tolerance_sec": tolerance_sec,
    }


def micro_timing_match_rate(
    total_matched: float,
    total_ref: float,
    total_pred: float,
) -> float:
    denom = max(float(total_pred), float(total_ref))
    if denom <= 0:
        return 0.0
    return float(total_matched) / denom


def timing_match_wrapper(
    pred_times: np.ndarray,
    ref_times: np.ndarray,
    tolerance_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    tol = float(tolerance_sec.reshape(-1)[0])
    n_matched, n_ref = timing_match_counts_numpy(
        pred_times,
        ref_times,
        tolerance_sec=tol,
    )
    n_pred = int(pred.size)
    return (
        np.array(n_matched, dtype=np.float64),
        np.array(n_ref, dtype=np.float64),
        np.array(n_pred, dtype=np.float64),
    )


def timing_match_teacher_wrapper(
    pred_times: np.ndarray,
    ref_times: np.ndarray,
    onset_mask: np.ndarray,
    tolerance_sec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(onset_mask, dtype=np.float64).reshape(-1) > 0.5
    pred_kept = np.asarray(pred_times, dtype=np.float64).reshape(-1)[mask]
    ref_kept = np.asarray(ref_times, dtype=np.float64).reshape(-1)[mask]
    return timing_match_wrapper(pred_kept, ref_kept, tolerance_sec)


# Backward-compatible aliases (AR overfit code).
ordered_onset_match_counts_numpy = timing_match_counts_numpy
ordered_onset_match_rate_numpy = timing_match_rate_numpy
