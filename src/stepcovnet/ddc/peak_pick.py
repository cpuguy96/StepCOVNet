"""Hamming peak-pick and ±20 ms F-scores for DDC placement (`donahue2017ddc`)."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import signal

from stepcovnet.ddc import constants


@dataclasses.dataclass(frozen=True)
class OnsetMatchCounts:
    """True/false counts for one chart or a pooled set of charts.

    Attributes:
        true_positives: Predicted peaks matched to a ground-truth step.
        false_positives: Predicted peaks with no match.
        false_negatives: Ground-truth steps with no predicted peak.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Return precision, or 0 when there are no predictions."""
        denom = self.true_positives + self.false_positives
        if denom <= 0:
            return 0.0
        return self.true_positives / denom

    @property
    def recall(self) -> float:
        """Return recall, or 0 when there is no ground truth."""
        denom = self.true_positives + self.false_negatives
        if denom <= 0:
            return 0.0
        return self.true_positives / denom

    @property
    def f_score(self) -> float:
        """Return the F1 score from precision and recall."""
        precision = self.precision
        recall = self.recall
        denom = precision + recall
        if denom <= 0:
            return 0.0
        return 2.0 * precision * recall / denom


def hamming_smooth(
    salience: np.ndarray,
    *,
    width: int = constants.HAMMING_WIDTH,
) -> np.ndarray:
    """Convolve a 1D salience trace with a Hamming window.

    Args:
        salience: Onset probabilities, shape ``(time,)``.
        width: Hamming window length in frames (DDC uses 5).

    Returns:
        Smoothed salience of the same length.

    Raises:
        ValueError: If ``width`` is not a positive odd integer.
    """
    if width < 1 or width % 2 == 0:
        raise ValueError(f"Hamming width must be a positive odd int, got {width}")
    flat = np.asarray(salience, dtype=np.float64).reshape(-1)
    window = np.hamming(width)
    return np.convolve(flat, window, mode="same")


def find_salience_peaks(
    salience: np.ndarray,
    *,
    threshold: float,
    width: int = constants.HAMMING_WIDTH,
) -> np.ndarray:
    """Return frame indices of Hamming-smoothed local maxima above ``threshold``.

    Matches `ddc_onset.find_peaks`: smooth with Hamming(5), take ``argrelextrema``,
    then keep peaks whose *unsmoothed* salience is at least ``threshold``.

    Args:
        salience: Onset probabilities, shape ``(time,)``.
        threshold: Minimum unsmoothed salience to keep a peak.
        width: Hamming window length in frames.

    Returns:
        Integer frame indices, sorted ascending.
    """
    flat = np.asarray(salience, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return np.zeros((0,), dtype=np.int64)
    smoothed = hamming_smooth(flat, width=width)
    peak_indices = signal.argrelextrema(smoothed, np.greater_equal, order=1)[0]
    kept = [int(index) for index in peak_indices if flat[index] >= threshold]
    return np.asarray(kept, dtype=np.int64)


def peak_times_sec(
    salience: np.ndarray,
    *,
    threshold: float,
    hop_sec: float = constants.HOP_SEC,
    width: int = constants.HAMMING_WIDTH,
) -> np.ndarray:
    """Return Hamming peak-pick times in seconds.

    Args:
        salience: Onset probabilities, shape ``(time,)``.
        threshold: Minimum unsmoothed salience to keep a peak.
        hop_sec: Frame hop in seconds.
        width: Hamming window length in frames.

    Returns:
        Sorted peak times in seconds.
    """
    frames = find_salience_peaks(salience, threshold=threshold, width=width)
    return (frames.astype(np.float64) * hop_sec).astype(np.float32)


def match_onsets(
    pred_times: np.ndarray,
    gt_times: np.ndarray,
    *,
    tolerance_sec: float = constants.ALIGN_TOLERANCE_SEC,
) -> OnsetMatchCounts:
    """Greedy 1-1 match of predicted times to ground truth within ``tolerance``.

    Args:
        pred_times: Predicted onset times in seconds.
        gt_times: Ground-truth onset times in seconds.
        tolerance_sec: Maximum |pred − gt| for a true positive (DDC: 20 ms).

    Returns:
        Match counts for this chart.

    Raises:
        ValueError: If ``tolerance_sec`` is negative.
    """
    if tolerance_sec < 0:
        raise ValueError(f"tolerance_sec must be non-negative, got {tolerance_sec}")
    pred = np.sort(np.asarray(pred_times, dtype=np.float64).reshape(-1))
    gt = np.sort(np.asarray(gt_times, dtype=np.float64).reshape(-1))
    if pred.size == 0:
        return OnsetMatchCounts(0, 0, int(gt.size))
    if gt.size == 0:
        return OnsetMatchCounts(0, int(pred.size), 0)
    used = np.zeros((gt.size,), dtype=bool)
    true_positives = 0
    for time_sec in pred:
        diffs = np.abs(gt - time_sec)
        diffs[used] = np.inf
        best = int(np.argmin(diffs))
        if np.isfinite(diffs[best]) and diffs[best] <= tolerance_sec:
            used[best] = True
            true_positives += 1
    false_positives = int(pred.size) - true_positives
    false_negatives = int(gt.size) - true_positives
    return OnsetMatchCounts(true_positives, false_positives, false_negatives)


def add_counts(left: OnsetMatchCounts, right: OnsetMatchCounts) -> OnsetMatchCounts:
    """Add two count objects fieldwise.

    Args:
        left: First counts.
        right: Second counts.

    Returns:
        Pooled counts.
    """
    return OnsetMatchCounts(
        left.true_positives + right.true_positives,
        left.false_positives + right.false_positives,
        left.false_negatives + right.false_negatives,
    )


def f_score_c(chart_scores: list[float]) -> float:
    """Return the chart-averaged F-score (DDC F-score_c).

    Args:
        chart_scores: Per-chart F1 values.

    Returns:
        Mean F1, or 0.0 when ``chart_scores`` is empty.
    """
    if not chart_scores:
        return 0.0
    return float(np.mean(np.asarray(chart_scores, dtype=np.float64)))


def f_score_m(counts: OnsetMatchCounts) -> float:
    """Return the micro-averaged F-score (DDC F-score_m).

    Args:
        counts: Pooled TP/FP/FN across charts.

    Returns:
        Micro F1 from pooled precision and recall.
    """
    return counts.f_score
