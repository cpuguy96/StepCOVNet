"""Val F-score_c / F-score_m (`M-ddc-20ms`) plus parallel ``timing_match``."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from stepcovnet import onset_null_baseline
from stepcovnet import timing_match as timing_match_lib
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
    """Pooled DDC placement metrics.

    Attributes:
        f_score_c: Chart-averaged F1 (``M-ddc-20ms``).
        f_score_m: Micro-averaged F1 (``M-ddc-20ms``).
        null_f_score_m: Density-matched ioi-shuffle F-score_m.
        timing_match: Micro ordered match @ 20 ms (not mixed into F-scores).
        null_timing_match: Ioi-shuffle ``timing_match`` at the same peak counts.
        timing_match_n_matched: Pooled ordered matches.
        timing_match_n_pred: Pooled predicted peak count.
        timing_match_n_ref: Pooled ground-truth onset count.
        timing_match_matched_count: Ordered match after keeping ``n_gt`` peaks.
        null_timing_match_matched_count: Ioi-shuffle ``timing_match`` at ``n_gt``.
        timing_match_matched_count_n_matched: Pooled matched-count ordered hits.
        timing_match_matched_count_n_pred: Pooled peaks kept at matched count.
        per_difficulty_threshold: Threshold chosen per difficulty.
        n_charts: Number of evaluated charts.
        counts: Pooled TP/FP/FN.
        charts: Per-chart rows.
    """

    f_score_c: float
    f_score_m: float
    null_f_score_m: float
    timing_match: float
    null_timing_match: float
    timing_match_n_matched: int
    timing_match_n_pred: int
    timing_match_n_ref: int
    timing_match_matched_count: float
    null_timing_match_matched_count: float
    timing_match_matched_count_n_matched: int
    timing_match_matched_count_n_pred: int
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
            "timing_match": self.timing_match,
            "null_timing_match": self.null_timing_match,
            "skill_timing_match": self.timing_match - self.null_timing_match,
            "timing_match_n_matched": self.timing_match_n_matched,
            "timing_match_n_pred": self.timing_match_n_pred,
            "timing_match_n_ref": self.timing_match_n_ref,
            "timing_match_n_denom": timing_match_lib.timing_match_denom(
                self.timing_match_n_pred,
                self.timing_match_n_ref,
            ),
            "timing_match_tolerance_sec": timing_match_lib.DEFAULT_TOLERANCE_SEC,
            "timing_match_matched_count": self.timing_match_matched_count,
            "null_timing_match_matched_count": self.null_timing_match_matched_count,
            "skill_timing_match_matched_count": (
                self.timing_match_matched_count - self.null_timing_match_matched_count
            ),
            "timing_match_matched_count_n_matched": (
                self.timing_match_matched_count_n_matched
            ),
            "timing_match_matched_count_n_pred": self.timing_match_matched_count_n_pred,
            "timing_match_matched_count_n_denom": timing_match_lib.timing_match_denom(
                self.timing_match_matched_count_n_pred,
                self.timing_match_n_ref,
            ),
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
        pred_times=pred_times,
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


def _chart_duration_sec(chart: datasets.PlacementChart) -> float:
    """Return a duration that covers both spectrogram length and last onset.

    Args:
        chart: Loaded chart.

    Returns:
        Duration in seconds.
    """
    duration = float(chart.n_frames * constants.HOP_SEC)
    if chart.gt_times.size:
        duration = max(duration, float(chart.gt_times[-1]) + constants.HOP_SEC)
    return duration


def _times_matched_count(
    pred_times: np.ndarray,
    salience: np.ndarray,
    n_keep: int,
    *,
    hop_sec: float = constants.HOP_SEC,
) -> np.ndarray:
    """Return up to ``n_keep`` peak times with the highest salience.

    Extra peaks are dropped; missing peaks are not invented. Times are sorted
    ascending so ``timing_match`` sees chronological order.

    Args:
        pred_times: Hamming peak-pick times in seconds.
        salience: Model probabilities, shape ``(time,)``.
        n_keep: Maximum peaks to keep (typically ``n_gt``).
        hop_sec: Frame hop used to index ``salience``.

    Returns:
        Sorted times of length ``min(n_pred, max(n_keep, 0))``.
    """
    times = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    keep = min(int(n_keep), int(times.size))
    if keep <= 0 or times.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if times.size <= keep:
        return np.sort(times).astype(np.float32)
    flat = np.asarray(salience, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return np.sort(times[:keep]).astype(np.float32)
    frames = np.clip(np.rint(times / hop_sec).astype(np.int64), 0, flat.size - 1)
    scores = flat[frames]
    order = np.lexsort((times, -scores))
    return np.sort(times[order[:keep]]).astype(np.float32)


def _null_metrics(
    charts: list[datasets.PlacementChart],
    n_preds: list[int],
    *,
    seed: int = 42,
) -> tuple[float, float]:
    """Density-matched ioi-shuffle floors for F-score_m and ``timing_match``.

    Both scores use the same shuffled times so the floors stay comparable.

    Args:
        charts: Evaluated charts.
        n_preds: Predicted peak counts aligned with ``charts``.
        seed: Null RNG seed.

    Returns:
        ``(null_f_score_m, null_timing_match)``.
    """
    rng = np.random.default_rng(seed)
    pooled = peak_pick.OnsetMatchCounts(0, 0, 0)
    total_matched = 0
    total_pred = 0
    total_ref = 0
    for chart, n_pred in zip(charts, n_preds, strict=True):
        null_times = onset_null_baseline.build_null_onsets(
            "ioi_shuffle",
            chart.gt_times,
            duration_sec=_chart_duration_sec(chart),
            n_pred=int(n_pred),
            rng=rng,
            hop_sec=constants.HOP_SEC,
        )
        pooled = peak_pick.add_counts(
            pooled,
            peak_pick.match_onsets(null_times, chart.gt_times),
        )
        n_matched, n_ref = timing_match_lib.timing_match_counts_numpy(
            null_times,
            chart.gt_times,
            tolerance_sec=timing_match_lib.DEFAULT_TOLERANCE_SEC,
        )
        total_matched += n_matched
        total_pred += int(null_times.size)
        total_ref += n_ref
    return pooled.f_score, timing_match_lib.micro_timing_match_rate(
        total_matched,
        total_ref,
        total_pred,
    )


def evaluate_placement(
    model: Any,
    charts: list[datasets.PlacementChart],
    *,
    thresholds: dict[str, float] | None = None,
    tune_on: bool = True,
    seed: int = 42,
) -> PlacementEvalReport:
    """Compute ``M-ddc-20ms`` F-scores plus parallel ``timing_match``.

    When ``tune_on`` is True, per-difficulty thresholds are chosen on the same
    charts (DDC tuned on val and reported test; this split has no test set).
    ``timing_match`` uses the same peak times; it is not mixed into F-score_c/m.
    ``timing_match_matched_count`` drops extra peaks to ``n_gt`` by salience
    (diagnostic only; literature Hamming POST is unchanged).

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
    total_matched = 0
    total_pred = 0
    total_ref = 0
    matched_count_hits = 0
    matched_count_pred = 0
    n_gts: list[int] = []
    for salience, chart in zip(saliences, charts, strict=True):
        threshold = thresholds.get(chart.difficulty, 0.5)
        row = _score_chart(salience, chart, threshold)
        rows.append(row)
        pooled = peak_pick.add_counts(pooled, row.counts)
        chart_f.append(row.counts.f_score)
        n_preds.append(row.n_pred)
        n_gts.append(row.n_gt)
        n_matched, n_ref = timing_match_lib.timing_match_counts_numpy(
            row.pred_times,
            chart.gt_times,
            tolerance_sec=timing_match_lib.DEFAULT_TOLERANCE_SEC,
        )
        total_matched += n_matched
        total_pred += row.n_pred
        total_ref += n_ref
        kept = _times_matched_count(row.pred_times, salience, row.n_gt)
        kept_matched, _ = timing_match_lib.timing_match_counts_numpy(
            kept,
            chart.gt_times,
            tolerance_sec=timing_match_lib.DEFAULT_TOLERANCE_SEC,
        )
        matched_count_hits += kept_matched
        matched_count_pred += int(kept.size)
    null_f_score_m, null_timing = _null_metrics(charts, n_preds, seed=seed)
    _, null_matched_count = _null_metrics(charts, n_gts, seed=seed)
    return PlacementEvalReport(
        f_score_c=peak_pick.f_score_c(chart_f),
        f_score_m=peak_pick.f_score_m(pooled),
        null_f_score_m=null_f_score_m,
        timing_match=timing_match_lib.micro_timing_match_rate(
            total_matched,
            total_ref,
            total_pred,
        ),
        null_timing_match=null_timing,
        timing_match_n_matched=total_matched,
        timing_match_n_pred=total_pred,
        timing_match_n_ref=total_ref,
        timing_match_matched_count=timing_match_lib.micro_timing_match_rate(
            matched_count_hits,
            total_ref,
            matched_count_pred,
        ),
        null_timing_match_matched_count=null_matched_count,
        timing_match_matched_count_n_matched=matched_count_hits,
        timing_match_matched_count_n_pred=matched_count_pred,
        per_difficulty_threshold=dict(thresholds),
        n_charts=len(charts),
        counts=pooled,
        charts=rows,
    )
