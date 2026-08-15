"""Val F-score_c / F-score_m for DDC placement (`M-ddc-20ms`)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from stepcovnet import onset_null_baseline
from stepcovnet.ddc import constants, datasets, features, peak_pick

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
    """

    song_key: str
    difficulty: str
    threshold: float
    counts: peak_pick.OnsetMatchCounts
    n_pred: int
    n_gt: int


@dataclasses.dataclass(frozen=True)
class PlacementEvalReport:
    """Pooled DDC placement metrics.

    Attributes:
        f_score_c: Chart-averaged F1.
        f_score_m: Micro-averaged F1.
        null_f_score_m: Density-matched ioi-shuffle F-score_m.
        per_difficulty_threshold: Threshold chosen per difficulty.
        n_charts: Number of evaluated charts.
        counts: Pooled TP/FP/FN.
        charts: Per-chart rows.
    """

    f_score_c: float
    f_score_m: float
    null_f_score_m: float
    per_difficulty_threshold: dict[str, float]
    n_charts: int
    counts: peak_pick.OnsetMatchCounts
    charts: list[ChartEvalResult]

    def as_dict(self) -> dict:
        """Serialize the report for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return {
            "f_score_c": self.f_score_c,
            "f_score_m": self.f_score_m,
            "null_f_score_m": self.null_f_score_m,
            "skill_f_score_m": self.f_score_m - self.null_f_score_m,
            "per_difficulty_threshold": dict(self.per_difficulty_threshold),
            "n_charts": self.n_charts,
            "true_positives": self.counts.true_positives,
            "false_positives": self.counts.false_positives,
            "false_negatives": self.counts.false_negatives,
            "paper_f_score_c": constants.PAPER_F_SCORE_C,
            "paper_f_score_m": constants.PAPER_F_SCORE_M,
        }


def predict_chart_salience(model: Any, chart: datasets.PlacementChart) -> np.ndarray:
    """Run the C-LSTM on one chart in GPU-safe frame chunks.

    LSTM state is reset at each chunk boundary (256 frames). That is a known
    deviation from DDC's carried BPTT state at eval.

    Args:
        model: Keras placement model.
        chart: Loaded chart with a log-mel spectrogram.

    Returns:
        Salience vector of shape ``(time,)``.
    """
    n_frames = chart.n_frames
    chunks: list[np.ndarray] = []
    for start in range(0, n_frames, constants.PREDICT_CHUNK_FRAMES):
        length = min(constants.PREDICT_CHUNK_FRAMES, n_frames - start)
        windows = features.context_windows_span(chart.spec, start, length)
        difficulty = np.broadcast_to(
            chart.difficulty_vec[np.newaxis, :],
            (length, constants.N_DIFFICULTIES),
        )
        prediction = model.predict(
            {
                "audio": windows[np.newaxis, ...],
                "difficulty": np.asarray(difficulty, dtype=np.float32)[np.newaxis, ...],
            },
            verbose=0,
        )
        chunks.append(np.asarray(prediction, dtype=np.float32).reshape(-1))
    return np.concatenate(chunks, axis=0)


def _score_chart(
    salience: np.ndarray,
    chart: datasets.PlacementChart,
    threshold: float,
) -> ChartEvalResult:
    """Score one chart at a fixed threshold.

    Args:
        salience: Model probabilities, shape ``(time,)``.
        chart: Loaded chart.
        threshold: Peak-pick threshold.

    Returns:
        Per-chart eval row.
    """
    pred_times = peak_pick.peak_times_sec(salience, threshold=threshold)
    counts = peak_pick.match_onsets(pred_times, chart.gt_times)
    return ChartEvalResult(
        song_key=chart.song_key,
        difficulty=chart.difficulty,
        threshold=threshold,
        counts=counts,
        n_pred=int(pred_times.size),
        n_gt=int(chart.gt_times.size),
    )


def choose_thresholds(
    saliences: list[np.ndarray],
    charts: list[datasets.PlacementChart],
    *,
    grid: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, float]:
    """Pick a per-difficulty threshold that maximizes micro F1 on ``charts``.

    Args:
        saliences: Model traces aligned with ``charts``.
        charts: Charts used for the sweep (typically the val split).
        grid: Candidate thresholds.

    Returns:
        Mapping difficulty → threshold. Missing difficulties default to 0.5.
    """
    chosen: dict[str, float] = {}
    labels = sorted({chart.difficulty for chart in charts})
    for label in labels:
        best_threshold = 0.5
        best_score = -1.0
        for threshold in grid:
            pooled = peak_pick.OnsetMatchCounts(0, 0, 0)
            for salience, chart in zip(saliences, charts, strict=True):
                if chart.difficulty != label:
                    continue
                row = _score_chart(salience, chart, threshold)
                pooled = peak_pick.add_counts(pooled, row.counts)
            score = pooled.f_score
            if score > best_score:
                best_score = score
                best_threshold = threshold
        chosen[label] = best_threshold
    return chosen


def _null_f_score_m(
    charts: list[datasets.PlacementChart],
    n_preds: list[int],
    *,
    seed: int = 42,
) -> float:
    """Density-matched ioi-shuffle F-score_m (audio-blind floor).

    Args:
        charts: Evaluated charts.
        n_preds: Predicted peak counts aligned with ``charts``.
        seed: Null RNG seed.

    Returns:
        Micro F1 of the ioi-shuffle baseline.
    """
    rng = np.random.default_rng(seed)
    pooled = peak_pick.OnsetMatchCounts(0, 0, 0)
    for chart, n_pred in zip(charts, n_preds, strict=True):
        duration = float(chart.n_frames * constants.HOP_SEC)
        if chart.gt_times.size:
            duration = max(
                duration,
                float(chart.gt_times[-1]) + constants.HOP_SEC,
            )
        null_times = onset_null_baseline.build_null_onsets(
            "ioi_shuffle",
            chart.gt_times,
            duration_sec=duration,
            n_pred=int(n_pred),
            rng=rng,
            hop_sec=constants.HOP_SEC,
        )
        pooled = peak_pick.add_counts(
            pooled,
            peak_pick.match_onsets(null_times, chart.gt_times),
        )
    return pooled.f_score


def evaluate_placement(
    model: Any,
    charts: list[datasets.PlacementChart],
    *,
    thresholds: dict[str, float] | None = None,
    tune_on: bool = True,
    seed: int = 42,
) -> PlacementEvalReport:
    """Compute F-score_c, F-score_m, and an audio-blind floor.

    When ``tune_on`` is True, per-difficulty thresholds are chosen on the same
    charts (DDC tuned on val and reported test; this split has no test set).

    Args:
        model: Keras placement model.
        charts: Charts to score.
        thresholds: Optional precomputed difficulty → threshold map.
        tune_on: When True and ``thresholds`` is None, sweep on ``charts``.
        seed: Null-baseline seed.

    Returns:
        Placement eval report.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    saliences = [predict_chart_salience(model, chart) for chart in charts]
    if thresholds is None:
        if tune_on:
            thresholds = choose_thresholds(saliences, charts)
        else:
            thresholds = {chart.difficulty: 0.5 for chart in charts}
    rows: list[ChartEvalResult] = []
    pooled = peak_pick.OnsetMatchCounts(0, 0, 0)
    chart_f: list[float] = []
    n_preds: list[int] = []
    for salience, chart in zip(saliences, charts, strict=True):
        threshold = thresholds.get(chart.difficulty, 0.5)
        row = _score_chart(salience, chart, threshold)
        rows.append(row)
        pooled = peak_pick.add_counts(pooled, row.counts)
        chart_f.append(row.counts.f_score)
        n_preds.append(row.n_pred)
    return PlacementEvalReport(
        f_score_c=peak_pick.f_score_c(chart_f),
        f_score_m=peak_pick.f_score_m(pooled),
        null_f_score_m=_null_f_score_m(charts, n_preds, seed=seed),
        per_difficulty_threshold=dict(thresholds),
        n_charts=len(charts),
        counts=pooled,
        charts=rows,
    )
