"""Audio-blind null baselines for onset timing metrics.

Hungarian event F1 at a 20 ms tolerance has a high chance floor when ground-truth
onsets are dense: charts average ~5.5 onsets/sec (~181 ms between onsets), so a
+/-20 ms match window already covers ~22% of the timeline. A predictor that never
listens to the audio and only emits the right *number* of onsets therefore scores
well above zero, and a raw F1 cannot be read as skill.

These baselines score such predictors with the same matcher the model is scored
with, at the same prediction count, so a reported number can be compared against
its own floor. ``skill_over_null`` normalizes a score against that floor.

Kinds (all use only onset count, song duration, and — for ``ioi_shuffle`` — the
multiset of ground-truth inter-onset intervals; none use audio):

* ``uniform_duration`` — draws uniformly over ``[0, duration_sec)``.
* ``regular_grid`` — a metronome evenly spanning the ground-truth support.
* ``ioi_shuffle`` — permutes the ground-truth inter-onset intervals.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from stepcovnet import timing_match
from stepcovnet.onset_events import metrics

DEFAULT_KINDS = ("uniform_duration", "regular_grid", "ioi_shuffle")
DEFAULT_SEED = 42


@dataclasses.dataclass(frozen=True)
class NullCounts:
    """Match counts for one audio-blind baseline on one song.

    Attributes:
        kind: Baseline name from :data:`DEFAULT_KINDS`.
        true_positives: Hungarian-matched predictions within tolerance.
        false_positives: Unmatched predictions.
        false_negatives: Unmatched ground-truth onsets.
        n_matched_ordered: Ordered ``timing_match`` hits.
        n_denom_ordered: Ordered ``timing_match`` denominator.
    """

    kind: str
    true_positives: int
    false_positives: int
    false_negatives: int
    n_matched_ordered: int
    n_denom_ordered: int


def build_null_onsets(
    kind: str,
    gt_times: np.ndarray,
    *,
    duration_sec: float,
    n_pred: int,
    rng: np.random.Generator,
    hop_sec: float = 0.02,
) -> np.ndarray:
    """Return audio-blind predicted onset times, sorted ascending.

    Args:
        kind: Baseline name from :data:`DEFAULT_KINDS`.
        gt_times: Ground-truth onset times in seconds, ascending.
        duration_sec: Song duration in seconds.
        n_pred: Number of onsets to emit (match the model being compared).
        rng: Seeded generator, so a baseline is reproducible.
        hop_sec: Prediction grid; emitted times snap to this, as model
            predictions do.

    Returns:
        Sorted onset times in seconds; empty when ``n_pred`` is not positive.

    Raises:
        ValueError: If ``kind`` is not a supported baseline.
    """
    gt = np.asarray(gt_times, dtype=np.float64).reshape(-1)
    n = int(n_pred)
    if n <= 0 or gt.size == 0 or duration_sec <= 0.0:
        return np.zeros(0, dtype=np.float64)
    lo = float(gt.min())
    hi = float(gt.max())
    if kind == "uniform_duration":
        times = rng.uniform(0.0, float(duration_sec), size=n)
    elif kind == "regular_grid":
        times = np.linspace(lo, hi, num=n) if hi > lo else np.full(n, lo)
    elif kind == "ioi_shuffle":
        iois = np.diff(gt)
        if iois.size == 0:
            times = np.full(n, lo)
        else:
            draw = rng.choice(iois, size=max(n - 1, 0), replace=True)
            times = lo + np.concatenate([[0.0], np.cumsum(draw)])
    else:
        raise ValueError(f"unsupported null baseline kind: {kind!r}")
    times = np.clip(times, 0.0, float(duration_sec))
    if hop_sec > 0.0:
        times = np.round(times / hop_sec) * hop_sec
    return np.sort(times)


def null_counts_for_song(
    gt_times: np.ndarray,
    *,
    duration_sec: float,
    n_pred: int,
    tolerance_sec: float,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    seed: int = DEFAULT_SEED,
    hop_sec: float = 0.02,
) -> list[NullCounts]:
    """Score every baseline on one song at the model's own prediction count.

    Args:
        gt_times: Ground-truth onset times in seconds, ascending.
        duration_sec: Song duration in seconds.
        n_pred: Prediction count the model emitted for this song.
        tolerance_sec: Match tolerance in seconds.
        kinds: Baselines to score.
        seed: Base seed; each kind gets a distinct derived stream.
        hop_sec: Prediction grid for emitted times.

    Returns:
        One :class:`NullCounts` per requested kind.
    """
    gt = np.asarray(gt_times, dtype=np.float64).reshape(-1)
    out: list[NullCounts] = []
    for offset, kind in enumerate(kinds):
        rng = np.random.default_rng(seed + 1000 * offset)
        pred = build_null_onsets(
            kind,
            gt,
            duration_sec=duration_sec,
            n_pred=n_pred,
            rng=rng,
            hop_sec=hop_sec,
        )
        if pred.size == 0 or gt.size == 0:
            out.append(
                NullCounts(kind, 0, int(pred.size), int(gt.size), 0, int(gt.size)),
            )
            continue
        tp, fp, fn = metrics.count_event_onset_errors_numpy(
            pred,
            np.ones_like(pred),
            gt,
            np.ones_like(gt),
            tolerance_sec,
            0.5,
            0.0,
        )
        n_matched, n_ref = timing_match.timing_match_counts_numpy(
            pred,
            gt,
            tolerance_sec=tolerance_sec,
        )
        out.append(
            NullCounts(
                kind=kind,
                true_positives=int(tp),
                false_positives=int(fp),
                false_negatives=int(fn),
                n_matched_ordered=int(n_matched),
                n_denom_ordered=timing_match.timing_match_denom(
                    int(pred.size),
                    int(n_ref),
                ),
            ),
        )
    return out


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom else 0.0


def aggregate_null_counts(
    rows: list[list[NullCounts]],
) -> dict[str, dict[str, float | int]]:
    """Micro-average per-song baseline counts across a split.

    Args:
        rows: Per-song lists returned by :func:`null_counts_for_song`.

    Returns:
        Mapping from baseline kind to its micro ``event_f1``, ``timing_match``,
        and raw counts.
    """
    by_kind: dict[str, list[NullCounts]] = {}
    for row in rows:
        for counts in row:
            by_kind.setdefault(counts.kind, []).append(counts)
    out: dict[str, dict[str, float | int]] = {}
    for kind, counts in by_kind.items():
        tp = sum(c.true_positives for c in counts)
        fp = sum(c.false_positives for c in counts)
        fn = sum(c.false_negatives for c in counts)
        matched = sum(c.n_matched_ordered for c in counts)
        denom = sum(c.n_denom_ordered for c in counts)
        out[kind] = {
            "event_f1": _f1_from_counts(tp, fp, fn),
            "timing_match": float(matched / denom) if denom else 0.0,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    return out


def skill_over_null(score: float, null_score: float) -> float:
    """Return how far ``score`` clears a chance floor, as a fraction of headroom.

    ``0`` means indistinguishable from the baseline, ``1`` means perfect, and a
    negative value means the baseline wins.

    Args:
        score: Measured metric value in ``[0, 1]``.
        null_score: Baseline value for the same metric and prediction count.
    """
    headroom = 1.0 - float(null_score)
    if headroom <= 0.0:
        return 0.0
    return (float(score) - float(null_score)) / headroom


def strongest_null(
    aggregated: dict[str, dict[str, float | int]],
    *,
    metric: str = "event_f1",
) -> tuple[str, float]:
    """Return the ``(kind, value)`` of the hardest baseline to beat.

    Args:
        aggregated: Output of :func:`aggregate_null_counts`.
        metric: Metric key to rank baselines by.
    """
    if not aggregated:
        return ("", 0.0)
    kind = max(aggregated, key=lambda k: float(aggregated[k][metric]))
    return (kind, float(aggregated[kind][metric]))
