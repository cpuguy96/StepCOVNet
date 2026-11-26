"""Zero-TF ``M-ddc-20ms`` evaluation for PyTorch DDC placement."""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from torch import nn

from stepcovnet import timing_match as timing_match_lib
from stepcovnet.ddc import constants, datasets, features, peak_pick
from stepcovnet.pytorch import runtime

DEFAULT_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))


@dataclasses.dataclass(frozen=True)
class ChartEvalResult:
    """Peak-pick scores for one chart.

    Attributes:
        song_key: ``bundle/id`` identifier.
        difficulty: DDR difficulty label.
        threshold: Peak-pick threshold used for this chart.
        counts: TP/FP/FN at that threshold.
        n_pred: Number of predicted peaks.
        n_gt: Number of ground-truth onsets.
        pred_times: Hamming peak-pick times in seconds.
    """

    song_key: str
    difficulty: str
    threshold: float
    counts: peak_pick.OnsetMatchCounts
    n_pred: int
    n_gt: int
    pred_times: np.ndarray


@dataclasses.dataclass(frozen=True)
class PlacementEvalReport:
    """Pooled DDC placement metrics (``M-ddc-20ms``).

    Attributes:
        f_score_c: Chart-averaged F1.
        f_score_m: Micro-averaged F1.
        null_f_score_m: Density-matched IOI-shuffle F-score_m.
        timing_match: Micro ordered match @ 20 ms.
        per_difficulty_threshold: Threshold chosen per difficulty.
        n_charts: Number of evaluated charts.
        counts: Pooled TP/FP/FN.
        charts: Per-chart rows.
    """

    f_score_c: float
    f_score_m: float
    null_f_score_m: float
    timing_match: float
    per_difficulty_threshold: dict[str, float]
    n_charts: int
    counts: peak_pick.OnsetMatchCounts
    charts: list[ChartEvalResult]

    def as_dict(self) -> dict[str, object]:
        """Serialize the report for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return {
            "f_score_c": self.f_score_c,
            "f_score_m": self.f_score_m,
            "null_f_score_m": self.null_f_score_m,
            "skill_f_score_m": self.f_score_m - self.null_f_score_m,
            "timing_match": self.timing_match,
            "per_difficulty_threshold": dict(self.per_difficulty_threshold),
            "n_charts": self.n_charts,
            "true_positives": self.counts.true_positives,
            "false_positives": self.counts.false_positives,
            "false_negatives": self.counts.false_negatives,
            "paper_f_score_c": constants.PAPER_F_SCORE_C,
            "paper_f_score_m": constants.PAPER_F_SCORE_M,
            "framework": "pytorch",
        }


def _ioi_shuffle_null(
    ref_times: np.ndarray,
    n_pred: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a density-matched IOI-shuffled null time sequence.

    Args:
        ref_times: Ground-truth onset times in seconds.
        n_pred: Target predicted count.
        rng: NumPy RNG.

    Returns:
        Sorted float64 null onset times.
    """
    ref = np.sort(np.asarray(ref_times, dtype=np.float64).reshape(-1))
    if n_pred <= 0 or ref.size == 0:
        return np.zeros(0, dtype=np.float64)
    if ref.size == 1:
        return np.full((n_pred,), float(ref[0]), dtype=np.float64)
    iois = np.diff(ref)
    if n_pred == 1:
        return np.array([float(ref[0])], dtype=np.float64)
    sampled_iois = rng.choice(iois, size=n_pred - 1, replace=True)
    return np.concatenate([[float(ref[0])], float(ref[0]) + np.cumsum(sampled_iois)])


@torch.no_grad()
def predict_chart_salience(
    model: nn.Module,
    chart: datasets.PlacementChart,
    *,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Run the PyTorch C-LSTM on one chart in GPU-safe frame chunks.

    Args:
        model: PyTorch DDC placement model.
        chart: Loaded chart with log-mel spectrogram.
        device: Target device (defaults to model's parameter device).
        amp_dtype: Optional autocast dtype (default fp32).

    Returns:
        Salience vector of shape ``(time,)``.
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
        n_frames = chart.n_frames
        chunks: list[np.ndarray] = []
        for start in range(0, n_frames, constants.PREDICT_CHUNK_FRAMES):
            length = min(constants.PREDICT_CHUNK_FRAMES, n_frames - start)
            windows = features.context_windows_span(chart.spec, start, length)
            difficulty = np.broadcast_to(
                chart.difficulty_vec[np.newaxis, :],
                (length, constants.N_DIFFICULTIES),
            )
            inputs = {
                "audio": torch.as_tensor(
                    windows[np.newaxis, ...], dtype=torch.float32, device=target_device
                ),
                "difficulty": torch.as_tensor(
                    np.asarray(difficulty, dtype=np.float32)[np.newaxis, ...],
                    dtype=torch.float32,
                    device=target_device,
                ),
            }
            with runtime.autocast_context(target_device, amp_dtype):
                pred = model(inputs)
            chunks.append(pred.detach().float().cpu().numpy().reshape(-1))
        return np.concatenate(chunks, axis=0)
    finally:
        if was_training:
            model.train()


def evaluate_placement(
    model: nn.Module,
    charts: list[datasets.PlacementChart],
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    tolerance_sec: float = constants.ALIGN_TOLERANCE_SEC,
    seed: int = 42,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> PlacementEvalReport:
    """Evaluate PyTorch DDC placement with per-difficulty Hamming peak-pick sweep.

    Args:
        model: Trained PyTorch C-LSTM model.
        charts: Validation charts.
        thresholds: Candidate peak-pick thresholds.
        tolerance_sec: Match tolerance in seconds (default 20 ms).
        seed: RNG seed for IOI-shuffle null baseline.
        device: Optional inference device.
        amp_dtype: Optional autocast dtype.

    Returns:
        Pooled :class:`PlacementEvalReport`.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    salience_by_chart = [
        predict_chart_salience(model, chart, device=device, amp_dtype=amp_dtype)
        for chart in charts
    ]
    by_diff: dict[str, list[int]] = {}
    for idx, chart in enumerate(charts):
        by_diff.setdefault(chart.difficulty, []).append(idx)

    best_thr_by_diff: dict[str, float] = {}
    for diff, indices in by_diff.items():
        best_thr = float(thresholds[0])
        best_score = -1.0
        for thr in thresholds:
            pooled = peak_pick.OnsetMatchCounts(0, 0, 0)
            for idx in indices:
                pred_times = peak_pick.peak_times_sec(
                    salience_by_chart[idx],
                    threshold=thr,
                )
                counts = peak_pick.match_onsets(
                    pred_times,
                    charts[idx].gt_times,
                    tolerance_sec=tolerance_sec,
                )
                pooled = peak_pick.add_counts(pooled, counts)
            if pooled.f_score > best_score:
                best_score = pooled.f_score
                best_thr = float(thr)
        best_thr_by_diff[diff] = best_thr

    rng = np.random.default_rng(seed)
    chart_results: list[ChartEvalResult] = []
    pooled_counts = peak_pick.OnsetMatchCounts(0, 0, 0)
    null_pooled = peak_pick.OnsetMatchCounts(0, 0, 0)
    tm_matched = 0
    tm_pred = 0
    tm_ref = 0
    for idx, chart in enumerate(charts):
        thr = best_thr_by_diff[chart.difficulty]
        pred_times = peak_pick.peak_times_sec(salience_by_chart[idx], threshold=thr)
        counts = peak_pick.match_onsets(
            pred_times,
            chart.gt_times,
            tolerance_sec=tolerance_sec,
        )
        pooled_counts = peak_pick.add_counts(pooled_counts, counts)
        null_times = _ioi_shuffle_null(chart.gt_times, int(pred_times.size), rng)
        null_counts = peak_pick.match_onsets(
            null_times,
            chart.gt_times,
            tolerance_sec=tolerance_sec,
        )
        null_pooled = peak_pick.add_counts(null_pooled, null_counts)
        matched_i, ref_i = timing_match_lib.timing_match_counts_numpy(
            pred_times,
            chart.gt_times,
            tolerance_sec=tolerance_sec,
        )
        tm_matched += matched_i
        tm_pred += int(pred_times.size)
        tm_ref += ref_i
        chart_results.append(
            ChartEvalResult(
                song_key=chart.song_key,
                difficulty=chart.difficulty,
                threshold=thr,
                counts=counts,
                n_pred=int(pred_times.size),
                n_gt=int(chart.gt_times.size),
                pred_times=pred_times,
            )
        )

    return PlacementEvalReport(
        f_score_c=peak_pick.f_score_c([row.counts.f_score for row in chart_results]),
        f_score_m=peak_pick.f_score_m(pooled_counts),
        null_f_score_m=peak_pick.f_score_m(null_pooled),
        timing_match=timing_match_lib.micro_timing_match_rate(
            tm_matched, tm_ref, tm_pred
        ),
        per_difficulty_threshold=best_thr_by_diff,
        n_charts=len(charts),
        counts=pooled_counts,
        charts=chart_results,
    )
