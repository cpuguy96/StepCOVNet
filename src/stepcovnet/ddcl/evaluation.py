"""``M-slot48`` binary F1 for DDCL placement (`omalley2025ddcl`, `omalley2026itgpt`).

Reports micro F1 at threshold 0.5 and the max-F1 threshold sweep used in the
ITGPT comparison table. This is **not** Hamming ±20 ms (`M-ddc-20ms`).
"""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import torch
from torch import nn

from stepcovnet import onset_null_baseline, timing_match
from stepcovnet.ddc import peak_pick
from stepcovnet.ddcl import constants, datasets, slots
from stepcovnet.onset_events import metrics

DEFAULT_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
CORRUPTION_KINDS = ("none", "shuffle", "zeros")
DIAGNOSTIC_CONDITIONS = (
    "oracle",
    "raw",
    "fp_removed",
    "fn_filled",
    "chart_top_k",
    "beat_top_k",
)


class _PredictProtocol(typing.Protocol):
    """Minimal duck-typed predict interface used by test stubs."""

    def predict(self, inputs: object, verbose: int = 0) -> object:
        """Return slot probabilities.

        Args:
            inputs: Model input dict.
            verbose: Unused verbosity flag.

        Returns:
            Array-like slot scores.
        """


SlotPredictor = nn.Module | _PredictProtocol


@dataclasses.dataclass(frozen=True)
class Slot48Counts:
    """Pooled binary counts on the 48-slot grid.

    Attributes:
        true_positives: Predicted slots matched to a ground-truth 1.
        false_positives: Predicted 1s with ground-truth 0.
        false_negatives: Ground-truth 1s with predicted 0.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def f_score(self) -> float:
        """Return binary F1 from the pooled counts."""
        return peak_pick.OnsetMatchCounts(
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            false_negatives=self.false_negatives,
        ).f_score


def counts_at_threshold(
    pred: np.ndarray, target: np.ndarray, threshold: float
) -> Slot48Counts:
    """Score one ``(n_beats, 48)`` pair at a decision threshold.

    Args:
        pred: Predicted slot probabilities.
        target: Ground-truth 0/1 slots.
        threshold: Decision threshold.

    Returns:
        Pooled counts.
    """
    pred_bin = np.asarray(pred, dtype=np.float64) >= float(threshold)
    tgt_bin = np.asarray(target, dtype=np.float64) >= 0.5
    true_positives = int(np.logical_and(pred_bin, tgt_bin).sum())
    false_positives = int(np.logical_and(pred_bin, np.logical_not(tgt_bin)).sum())
    false_negatives = int(np.logical_and(np.logical_not(pred_bin), tgt_bin).sum())
    return Slot48Counts(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def shuffle_slot_null(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute beats of the GT slot matrix (density-matched, audio-blind).

    Args:
        target: Ground-truth ``(n_beats, n_slots)``.
        rng: NumPy generator.

    Returns:
        Shuffled copy.
    """
    out = np.asarray(target, dtype=np.float32).copy()
    order = rng.permutation(out.shape[0])
    return out[order]


@dataclasses.dataclass(frozen=True)
class Slot48EvalReport:
    """Pooled ``M-slot48`` scores.

    Attributes:
        f1_at_05: Micro F1 at threshold 0.5.
        f1_max: Best micro F1 over the threshold sweep.
        best_threshold: Threshold that achieved ``f1_max``.
        null_f1_at_05: Beat-shuffled GT scored at 0.5 against real GT.
        n_charts: Charts evaluated.
        n_beats: Pooled beat count.
        counts_at_05: Pooled counts at 0.5.
    """

    f1_at_05: float
    f1_max: float
    best_threshold: float
    null_f1_at_05: float
    n_charts: int
    n_beats: int
    counts_at_05: Slot48Counts

    def as_dict(self) -> dict:
        """Serialize for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return {
            "metric": "M-slot48",
            "f1_at_05": self.f1_at_05,
            "f1_max": self.f1_max,
            "best_threshold": self.best_threshold,
            "null_f1_at_05": self.null_f1_at_05,
            "skill_f1_at_05": self.f1_at_05 - self.null_f1_at_05,
            "n_charts": self.n_charts,
            "n_beats": self.n_beats,
            "true_positives": self.counts_at_05.true_positives,
            "false_positives": self.counts_at_05.false_positives,
            "false_negatives": self.counts_at_05.false_negatives,
            "published_f1_at_05_expanded_fraxtil": (
                constants.PUBLISHED_F1_AT_05_EXPANDED
            ),
            "published_f1_max_expanded_fraxtil": constants.PUBLISHED_F1_MAX_EXPANDED,
            "citation": "omalley2025ddcl",
        }


def evaluate_slot48(
    model: SlotPredictor,
    charts: list[datasets.DdclChart],
    *,
    seed: int = 42,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> Slot48EvalReport:
    """Predict every val chart and pool ``M-slot48``.

    Args:
        model: Keras model with DDCL placement inputs.
        charts: Loaded beat-grid charts.
        seed: Null-shuffle seed.
        thresholds: Sweep for max-F1.

    Returns:
        Pooled report.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    rng = np.random.default_rng(seed)
    pred_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    for chart in charts:
        pred_parts.append(predict_chart_slots(model, chart))
        tgt_parts.append(chart.slots)
    pred_all = np.concatenate(pred_parts, axis=0)
    tgt_all = np.concatenate(tgt_parts, axis=0)
    counts_05 = counts_at_threshold(pred_all, tgt_all, constants.THRESHOLD_05)
    best_threshold = constants.THRESHOLD_05
    best_f1 = counts_05.f_score
    for threshold in thresholds:
        score = counts_at_threshold(pred_all, tgt_all, threshold).f_score
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    null_pred = shuffle_slot_null(tgt_all, rng)
    null_counts = counts_at_threshold(null_pred, tgt_all, constants.THRESHOLD_05)
    return Slot48EvalReport(
        f1_at_05=counts_05.f_score,
        f1_max=best_f1,
        best_threshold=best_threshold,
        null_f1_at_05=null_counts.f_score,
        n_charts=len(charts),
        n_beats=int(tgt_all.shape[0]),
        counts_at_05=counts_05,
    )


@dataclasses.dataclass(frozen=True)
class DecodedTimingReport:
    """Ordered ``timing_match`` of slot-center times. Not ``M-slot48`` F1.

    Attributes:
        micro_timing_match: ``n_matched / max(n_pred, n_ref)``.
        tolerance_sec: Match window in seconds.
        threshold: Occupancy cutoff used to decode slots.
        n_matched: Ordered pairs inside the window.
        n_pred: Decoded onset count.
        n_ref: Chart onset count.
        n_charts: Charts pooled.
    """

    micro_timing_match: float
    tolerance_sec: float
    threshold: float
    n_matched: int
    n_pred: int
    n_ref: int
    n_charts: int

    def as_dict(self) -> dict[str, float | int | str]:
        """Serialize for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return {
            "metric": "timing_match",
            "tolerance_sec": self.tolerance_sec,
            "threshold": self.threshold,
            "micro_timing_match": self.micro_timing_match,
            "n_matched": self.n_matched,
            "n_pred": self.n_pred,
            "n_ref": self.n_ref,
            "n_charts": self.n_charts,
        }


def timing_match_from_predictions(
    charts: list[datasets.DdclChart],
    predictions: list[np.ndarray],
    *,
    threshold: float = constants.THRESHOLD_05,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
) -> DecodedTimingReport:
    """Score slot-center times against chart onset times.

    Args:
        charts: Charts with BPM segments and ``gt_times_sec``.
        predictions: Slot probabilities aligned with ``charts``.
        threshold: Occupancy cutoff for decoding.
        tolerance_sec: Ordered-match window.

    Returns:
        Micro ``timing_match`` and pooled counts. This is not ``M-slot48`` F1.

    Raises:
        ValueError: If ``charts`` is empty, lengths differ, or BPM is missing.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    if len(predictions) != len(charts):
        raise ValueError("predictions must align with charts")
    total_matched = 0
    total_pred = 0
    total_ref = 0
    for chart, pred in zip(charts, predictions, strict=True):
        if not chart.bpm_segments:
            raise ValueError(f"missing BPM segments for {chart.song_key}")
        decoded = slots.occupied_center_times(
            pred,
            chart.offset_sec,
            chart.bpm_segments,
            threshold=threshold,
        )
        ref = np.asarray(chart.gt_times_sec, dtype=np.float64).reshape(-1)
        n_matched, n_ref = timing_match.timing_match_counts_numpy(
            decoded,
            ref,
            tolerance_sec=tolerance_sec,
        )
        total_matched += n_matched
        total_pred += int(decoded.size)
        total_ref += n_ref
    denom = timing_match.timing_match_denom(total_pred, total_ref)
    micro = float(total_matched) / float(denom) if denom else 0.0
    return DecodedTimingReport(
        micro_timing_match=micro,
        tolerance_sec=float(tolerance_sec),
        threshold=float(threshold),
        n_matched=total_matched,
        n_pred=total_pred,
        n_ref=total_ref,
        n_charts=len(charts),
    )


def evaluate_decoded_timing(
    model: SlotPredictor,
    charts: list[datasets.DdclChart],
    *,
    threshold: float = constants.THRESHOLD_05,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
) -> DecodedTimingReport:
    """Predict slots and score decoded times with ``timing_match``.

    Args:
        model: Keras model with DDCL placement inputs.
        charts: Loaded beat-grid charts with BPM and onset times.
        threshold: Occupancy cutoff for decoding.
        tolerance_sec: Ordered-match window.

    Returns:
        Micro ``timing_match`` report.
    """
    predictions = [predict_chart_slots(model, chart) for chart in charts]
    return timing_match_from_predictions(
        charts,
        predictions,
        threshold=threshold,
        tolerance_sec=tolerance_sec,
    )


def predict_chart_slots(model: SlotPredictor, chart: datasets.DdclChart) -> np.ndarray:
    """Return slot probabilities for one chart.

    Args:
        model: Trained PyTorch placement model or duck-typed predictor stub.
        chart: Loaded chart.

    Returns:
        Probabilities ``(n_beats, 48)``.
    """
    parts: list[np.ndarray] = []
    if isinstance(model, nn.Module):
        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for beat_idxs in datasets.iter_window_batches(chart):
                    inputs_np = datasets.windows_for_indices(chart, beat_idxs)
                    inputs_t = {
                        k: torch.as_tensor(v, dtype=torch.float32, device=device)
                        for k, v in inputs_np.items()
                    }
                    pred = model(inputs_t).detach().float().cpu().numpy()
                    if pred.ndim == 1:
                        pred = pred.reshape(1, -1)
                    parts.append(pred)
        finally:
            if was_training:
                model.train()
    else:
        for beat_idxs in datasets.iter_window_batches(chart):
            inputs = datasets.windows_for_indices(chart, beat_idxs)
            pred = np.asarray(model.predict(inputs, verbose=0), dtype=np.float32)
            if pred.ndim == 1:
                pred = pred.reshape(1, -1)
            parts.append(pred)
    return np.concatenate(parts, axis=0)


def _top_k_mask_flat(scores: np.ndarray, count: int) -> np.ndarray:
    """Select the ``count`` highest-probability cells across a chart grid.

    Args:
        scores: Slot probability matrix ``(n_beats, n_slots)``.
        count: Number of occupied cells to select.

    Returns:
        Binary float64 grid of shape ``scores.shape``.
    """
    grid = np.asarray(scores, dtype=np.float64)
    out = np.zeros_like(grid, dtype=np.float64)
    k = min(max(int(count), 0), int(grid.size))
    if k <= 0:
        return out
    flat = grid.reshape(-1)
    top_indices = np.argsort(-flat, kind="stable")[:k]
    out.reshape(-1)[top_indices] = 1.0
    return out


def _top_k_mask_per_beat(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Select the oracle count of highest-probability slots within each beat.

    Args:
        scores: Slot probability matrix ``(n_beats, n_slots)``.
        target: Ground-truth binary matrix ``(n_beats, n_slots)``.

    Returns:
        Binary float64 grid matching each beat's ground-truth slot count.
    """
    grid = np.asarray(scores, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64) >= 0.5
    out = np.zeros_like(grid, dtype=np.float64)
    for beat_idx in range(grid.shape[0]):
        k = int(tgt[beat_idx].sum())
        if k <= 0:
            continue
        row = grid[beat_idx]
        top_indices = np.argsort(-row, kind="stable")[: min(k, row.size)]
        out[beat_idx, top_indices] = 1.0
    return out


def _mean_abs_beat_rank_drift(pred_bin: np.ndarray, tgt_bin: np.ndarray) -> float:
    """Compute mean absolute cumulative count error across beats of one chart.

    Args:
        pred_bin: Predicted binary grid ``(n_beats, n_slots)``.
        tgt_bin: Ground-truth binary grid ``(n_beats, n_slots)``.

    Returns:
        Mean ``|cumsum(pred_per_beat - gt_per_beat)|`` over beats.
    """
    if pred_bin.shape[0] == 0:
        return 0.0
    diff_per_beat = pred_bin.astype(np.int64).sum(axis=1) - tgt_bin.astype(
        np.int64
    ).sum(axis=1)
    cum_drift = np.cumsum(diff_per_beat)
    return float(np.mean(np.abs(cum_drift)))


def _event_counts_at_tolerance(
    pred_times: np.ndarray,
    ref_times: np.ndarray,
    *,
    tolerance_sec: float,
) -> tuple[int, int, int]:
    """Count bipartite onset matches within ``tolerance_sec`` for one chart.

    Args:
        pred_times: Sorted predicted onset times in seconds.
        ref_times: Sorted reference onset times in seconds.
        tolerance_sec: Match window in seconds.

    Returns:
        Tuple ``(true_positives, false_positives, false_negatives)``.
    """
    pred = np.asarray(pred_times, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_times, dtype=np.float64).reshape(-1)
    if pred.size == 0 or ref.size == 0:
        return 0, int(pred.size), int(ref.size)
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pred,
        np.ones_like(pred),
        ref,
        np.ones_like(ref),
        tolerance_sec,
        0.5,
        0.0,
    )
    return int(tp), int(fp), int(fn)


@dataclasses.dataclass(frozen=True)
class MaskTimingConditionReport:
    """Pooled slot, rank-ordered, and bipartite timing metrics for one mask mode.

    Attributes:
        condition: Mask condition name from :data:`DIAGNOSTIC_CONDITIONS`.
        slot_f1: Micro ``M-slot48`` F1 for this binary mask.
        slot_tp: Pooled slot true positives.
        slot_fp: Pooled slot false positives.
        slot_fn: Pooled slot false negatives.
        micro_timing_match: Rank-indexed ``n_matched / max(n_pred, n_ref)``.
        timing_n_matched: Ordered pairs within ``tolerance_sec``.
        n_pred: Pooled decoded onset count.
        n_ref: Pooled ground-truth onset count.
        null_timing_match: Strongest audio-blind ``timing_match`` floor.
        null_timing_kind: Baseline kind achieving ``null_timing_match``.
        skill_timing_match: Normalized headroom skill over ``null_timing_match``.
        micro_event_f1: Bipartite Hungarian event F1 within ``tolerance_sec``.
        event_tp: Pooled bipartite true positives.
        event_fp: Pooled bipartite false positives.
        event_fn: Pooled bipartite false negatives.
        null_event_f1: Strongest audio-blind bipartite event-F1 floor.
        null_event_kind: Baseline kind achieving ``null_event_f1``.
        skill_event_f1: Normalized headroom skill over ``null_event_f1``.
        charts_exact_count: Charts where ``n_pred == n_ref``.
        mean_abs_beat_rank_drift: Mean across charts of per-beat ``|cumsum(FP - FN)|``.
    """

    condition: str
    slot_f1: float
    slot_tp: int
    slot_fp: int
    slot_fn: int
    micro_timing_match: float
    timing_n_matched: int
    n_pred: int
    n_ref: int
    null_timing_match: float
    null_timing_kind: str
    skill_timing_match: float
    micro_event_f1: float
    event_tp: int
    event_fp: int
    event_fn: int
    null_event_f1: float
    null_event_kind: str
    skill_event_f1: float
    charts_exact_count: int
    mean_abs_beat_rank_drift: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Serialize condition metrics for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ThresholdTimingRow:
    """Pooled slot, ordered-timing, and bipartite metrics at one cutoff.

    Attributes:
        threshold: Occupancy cutoff applied to slot probabilities.
        slot_f1: Micro ``M-slot48`` F1 at ``threshold``.
        micro_timing_match: Rank-indexed ``timing_match`` at ``threshold``.
        micro_event_f1: Bipartite Hungarian event F1 at ``threshold``.
        timing_n_matched: Ordered pairs within ``tolerance_sec``.
        n_pred: Pooled decoded onset count.
        n_ref: Pooled ground-truth onset count.
        charts_exact_count: Charts where ``n_pred == n_ref``.
        mean_abs_beat_rank_drift: Mean across charts of per-beat ``|cumsum(FP - FN)|``.
    """

    threshold: float
    slot_f1: float
    micro_timing_match: float
    micro_event_f1: float
    timing_n_matched: int
    n_pred: int
    n_ref: int
    charts_exact_count: int
    mean_abs_beat_rank_drift: float

    def as_dict(self) -> dict[str, float | int]:
        """Serialize threshold row for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DecodedTimingDiagnosticReport:
    """Counterfactual mask ablations and threshold sweep for decoded slot timing.

    Attributes:
        threshold: Primary occupancy cutoff for ``raw``, ``fp_removed``, ``fn_filled``.
        tolerance_sec: Time match tolerance in seconds.
        n_charts: Charts evaluated.
        n_beats: Pooled integer beats across ``n_charts``.
        conditions: Mapping from condition name to its pooled report.
        threshold_sweep: Sweep rows across occupancy cutoffs.
    """

    threshold: float
    tolerance_sec: float
    n_charts: int
    n_beats: int
    conditions: dict[str, MaskTimingConditionReport]
    threshold_sweep: tuple[ThresholdTimingRow, ...]

    def as_dict(self) -> dict[str, object]:
        """Serialize full diagnostic report for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return {
            "metric": "timing_match_diagnostic",
            "threshold": self.threshold,
            "tolerance_sec": self.tolerance_sec,
            "n_charts": self.n_charts,
            "n_beats": self.n_beats,
            "conditions": {
                name: cond.as_dict() for name, cond in self.conditions.items()
            },
            "threshold_sweep": [row.as_dict() for row in self.threshold_sweep],
        }


def _score_binary_masks(
    condition: str,
    charts: list[datasets.DdclChart],
    masks: list[np.ndarray],
    *,
    tolerance_sec: float,
    seed: int,
) -> MaskTimingConditionReport:
    """Pool slot, ordered-timing, bipartite, and null metrics for binary masks.

    Args:
        condition: Condition label.
        charts: Loaded charts with BPM segments and ``gt_times_sec``.
        masks: Binary ``(n_beats, 48)`` grids aligned with ``charts``.
        tolerance_sec: Match tolerance in seconds.
        seed: Base RNG seed for audio-blind null baselines.

    Returns:
        Pooled condition report.
    """
    slot_tp = 0
    slot_fp = 0
    slot_fn = 0
    timing_matched = 0
    total_pred = 0
    total_ref = 0
    event_tp = 0
    event_fp = 0
    event_fn = 0
    exact_count = 0
    drifts: list[float] = []
    null_rows: list[list[onset_null_baseline.NullCounts]] = []

    for chart_idx, (chart, mask) in enumerate(zip(charts, masks, strict=True)):
        pred_bin = np.asarray(mask, dtype=np.float64) >= 0.5
        tgt_bin = np.asarray(chart.slots, dtype=np.float64) >= 0.5
        counts = counts_at_threshold(pred_bin.astype(np.float64), chart.slots, 0.5)
        slot_tp += counts.true_positives
        slot_fp += counts.false_positives
        slot_fn += counts.false_negatives
        drifts.append(_mean_abs_beat_rank_drift(pred_bin, tgt_bin))

        decoded = slots.occupied_center_times(
            pred_bin.astype(np.float64),
            chart.offset_sec,
            chart.bpm_segments,
            threshold=0.5,
        )
        ref = np.asarray(chart.gt_times_sec, dtype=np.float64).reshape(-1)
        n_matched, n_ref = timing_match.timing_match_counts_numpy(
            decoded,
            ref,
            tolerance_sec=tolerance_sec,
        )
        n_pred = int(decoded.size)
        if n_pred == n_ref:
            exact_count += 1
        timing_matched += n_matched
        total_pred += n_pred
        total_ref += n_ref

        c_tp, c_fp, c_fn = _event_counts_at_tolerance(
            decoded,
            ref,
            tolerance_sec=tolerance_sec,
        )
        event_tp += c_tp
        event_fp += c_fp
        event_fn += c_fn

        end_sec = float(
            slots.beat_to_time_sec(
                float(chart.slots.shape[0]),
                chart.offset_sec,
                chart.bpm_segments,
            )
        )
        duration_sec = max(end_sec, float(ref.max()) if ref.size else 0.0)
        null_rows.append(
            onset_null_baseline.null_counts_for_song(
                ref,
                duration_sec=duration_sec,
                n_pred=n_pred,
                tolerance_sec=tolerance_sec,
                seed=seed + chart_idx,
            )
        )

    slot_f1 = Slot48Counts(slot_tp, slot_fp, slot_fn).f_score
    denom = timing_match.timing_match_denom(total_pred, total_ref)
    micro_timing = float(timing_matched) / float(denom) if denom else 0.0
    micro_event = peak_pick.OnsetMatchCounts(event_tp, event_fp, event_fn).f_score
    aggregated_nulls = onset_null_baseline.aggregate_null_counts(null_rows)
    null_event_kind, null_event_val = onset_null_baseline.strongest_null(
        aggregated_nulls,
        metric="event_f1",
    )
    null_timing_kind, null_timing_val = onset_null_baseline.strongest_null(
        aggregated_nulls,
        metric="timing_match",
    )
    return MaskTimingConditionReport(
        condition=condition,
        slot_f1=slot_f1,
        slot_tp=slot_tp,
        slot_fp=slot_fp,
        slot_fn=slot_fn,
        micro_timing_match=micro_timing,
        timing_n_matched=timing_matched,
        n_pred=total_pred,
        n_ref=total_ref,
        null_timing_match=null_timing_val,
        null_timing_kind=null_timing_kind,
        skill_timing_match=onset_null_baseline.skill_over_null(
            micro_timing,
            null_timing_val,
        ),
        micro_event_f1=micro_event,
        event_tp=event_tp,
        event_fp=event_fp,
        event_fn=event_fn,
        null_event_f1=null_event_val,
        null_event_kind=null_event_kind,
        skill_event_f1=onset_null_baseline.skill_over_null(
            micro_event,
            null_event_val,
        ),
        charts_exact_count=exact_count,
        mean_abs_beat_rank_drift=float(np.mean(drifts)) if drifts else 0.0,
    )


def diagnose_timing_from_predictions(
    charts: list[datasets.DdclChart],
    predictions: list[np.ndarray],
    *,
    threshold: float = constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
    seed: int = onset_null_baseline.DEFAULT_SEED,
) -> DecodedTimingDiagnosticReport:
    """Measure how false slots and missed slots affect decoded ``timing_match``.

    Evaluates six counterfactual binary masks on the same predicted probabilities
    (``oracle``, ``raw``, ``fp_removed``, ``fn_filled``, ``chart_top_k``,
    ``beat_top_k``) plus a threshold sweep, reporting both rank-indexed
    ``timing_match`` and order-invariant bipartite event F1 within ``tolerance_sec``.

    Args:
        charts: Loaded charts with BPM segments and ``gt_times_sec``.
        predictions: Predicted ``(n_beats, 48)`` slot probabilities per chart.
        threshold: Primary occupancy cutoff for ``raw``, ``fp_removed``, ``fn_filled``.
        thresholds: Occupancy cutoffs for the threshold sweep.
        tolerance_sec: Match window in seconds.
        seed: Base RNG seed for audio-blind null baselines.

    Returns:
        Counterfactual mask and threshold diagnostic report.

    Raises:
        ValueError: If ``charts`` is empty, lengths differ, or BPM is missing.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    if len(predictions) != len(charts):
        raise ValueError("predictions must align with charts")
    for chart in charts:
        if not chart.bpm_segments:
            raise ValueError(f"missing BPM segments for {chart.song_key}")

    oracle_masks: list[np.ndarray] = []
    raw_masks: list[np.ndarray] = []
    fp_removed_masks: list[np.ndarray] = []
    fn_filled_masks: list[np.ndarray] = []
    chart_top_k_masks: list[np.ndarray] = []
    beat_top_k_masks: list[np.ndarray] = []

    for chart, pred in zip(charts, predictions, strict=True):
        tgt_bin = np.asarray(chart.slots, dtype=np.float64) >= 0.5
        pred_probs = np.asarray(pred, dtype=np.float64)
        pred_bin = pred_probs >= float(threshold)
        oracle_masks.append(tgt_bin.astype(np.float64))
        raw_masks.append(pred_bin.astype(np.float64))
        fp_removed_masks.append(np.logical_and(pred_bin, tgt_bin).astype(np.float64))
        fn_filled_masks.append(np.logical_or(pred_bin, tgt_bin).astype(np.float64))
        chart_top_k_masks.append(_top_k_mask_flat(pred_probs, int(tgt_bin.sum())))
        beat_top_k_masks.append(_top_k_mask_per_beat(pred_probs, chart.slots))

    condition_masks = {
        "oracle": oracle_masks,
        "raw": raw_masks,
        "fp_removed": fp_removed_masks,
        "fn_filled": fn_filled_masks,
        "chart_top_k": chart_top_k_masks,
        "beat_top_k": beat_top_k_masks,
    }
    conditions = {
        name: _score_binary_masks(
            name,
            charts,
            masks,
            tolerance_sec=tolerance_sec,
            seed=seed,
        )
        for name, masks in condition_masks.items()
    }

    sweep_rows: list[ThresholdTimingRow] = []
    for tau in thresholds:
        slot_tp = 0
        slot_fp = 0
        slot_fn = 0
        timing_matched = 0
        total_pred = 0
        total_ref = 0
        event_tp = 0
        event_fp = 0
        event_fn = 0
        exact_count = 0
        drifts: list[float] = []
        for chart, pred in zip(charts, predictions, strict=True):
            pred_bin = np.asarray(pred, dtype=np.float64) >= float(tau)
            tgt_bin = np.asarray(chart.slots, dtype=np.float64) >= 0.5
            c_slot = counts_at_threshold(pred, chart.slots, float(tau))
            slot_tp += c_slot.true_positives
            slot_fp += c_slot.false_positives
            slot_fn += c_slot.false_negatives
            drifts.append(_mean_abs_beat_rank_drift(pred_bin, tgt_bin))
            decoded = slots.occupied_center_times(
                pred,
                chart.offset_sec,
                chart.bpm_segments,
                threshold=float(tau),
            )
            ref = np.asarray(chart.gt_times_sec, dtype=np.float64).reshape(-1)
            n_matched, n_ref = timing_match.timing_match_counts_numpy(
                decoded,
                ref,
                tolerance_sec=tolerance_sec,
            )
            n_pred = int(decoded.size)
            if n_pred == n_ref:
                exact_count += 1
            timing_matched += n_matched
            total_pred += n_pred
            total_ref += n_ref
            c_tp, c_fp, c_fn = _event_counts_at_tolerance(
                decoded,
                ref,
                tolerance_sec=tolerance_sec,
            )
            event_tp += c_tp
            event_fp += c_fp
            event_fn += c_fn
        denom = timing_match.timing_match_denom(total_pred, total_ref)
        sweep_rows.append(
            ThresholdTimingRow(
                threshold=float(tau),
                slot_f1=Slot48Counts(slot_tp, slot_fp, slot_fn).f_score,
                micro_timing_match=(
                    float(timing_matched) / float(denom) if denom else 0.0
                ),
                micro_event_f1=peak_pick.OnsetMatchCounts(
                    event_tp, event_fp, event_fn
                ).f_score,
                timing_n_matched=timing_matched,
                n_pred=total_pred,
                n_ref=total_ref,
                charts_exact_count=exact_count,
                mean_abs_beat_rank_drift=float(np.mean(drifts)) if drifts else 0.0,
            )
        )

    n_beats = sum(int(chart.slots.shape[0]) for chart in charts)
    return DecodedTimingDiagnosticReport(
        threshold=float(threshold),
        tolerance_sec=float(tolerance_sec),
        n_charts=len(charts),
        n_beats=n_beats,
        conditions=conditions,
        threshold_sweep=tuple(sweep_rows),
    )


def diagnose_decoded_timing(
    model: SlotPredictor,
    charts: list[datasets.DdclChart],
    *,
    threshold: float = constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
    seed: int = onset_null_baseline.DEFAULT_SEED,
) -> DecodedTimingDiagnosticReport:
    """Predict chart slots once and run counterfactual timing diagnostics.

    Args:
        model: Keras model with DDCL placement inputs.
        charts: Loaded beat-grid charts with BPM and onset times.
        threshold: Primary occupancy cutoff for ``raw``, ``fp_removed``, ``fn_filled``.
        thresholds: Occupancy cutoffs for the threshold sweep.
        tolerance_sec: Match window in seconds.
        seed: Base RNG seed for audio-blind null baselines.

    Returns:
        Counterfactual mask and threshold diagnostic report.
    """
    predictions = [predict_chart_slots(model, chart) for chart in charts]
    return diagnose_timing_from_predictions(
        charts,
        predictions,
        threshold=threshold,
        thresholds=thresholds,
        tolerance_sec=tolerance_sec,
        seed=seed,
    )


def corrupt_chart_audio(
    chart: datasets.DdclChart,
    corruption: str,
    *,
    seed: int = onset_null_baseline.DEFAULT_SEED,
) -> datasets.DdclChart:
    """Return a chart copy with ``beat_audio`` corrupted and ``stream`` intact.

    Args:
        chart: Source beat-grid chart.
        corruption: One of :data:`CORRUPTION_KINDS` (``none``, ``shuffle``, ``zeros``).
        seed: RNG seed used when ``corruption == "shuffle"``.

    Returns:
        Chart with modified ``beat_audio`` and identical metadata/targets.

    Raises:
        ValueError: If ``corruption`` is not supported.
    """
    if corruption == "none":
        return chart
    if corruption == "zeros":
        return dataclasses.replace(
            chart,
            beat_audio=np.zeros_like(chart.beat_audio),
        )
    if corruption == "shuffle":
        rng = np.random.default_rng(seed)
        order = rng.permutation(chart.beat_audio.shape[0])
        return dataclasses.replace(
            chart,
            beat_audio=chart.beat_audio[order].copy(),
        )
    raise ValueError(f"unsupported corruption kind: {corruption!r}")


@dataclasses.dataclass(frozen=True)
class AudioGroundingConditionReport:
    """Slot, decoded ±20 ms, and prediction-shift metrics for one corruption mode.

    Attributes:
        corruption: Corruption kind from :data:`CORRUPTION_KINDS`.
        slot_f1_at_05: Micro ``M-slot48`` F1 at ``threshold``.
        slot_f1_max: Best micro ``M-slot48`` F1 over the threshold sweep.
        slot_best_threshold: Threshold achieving ``slot_f1_max``.
        slot_null_f1_at_05: Beat-shuffled GT ``M-slot48`` F1 at ``threshold``.
        slot_skill_f1_at_05: Raw ``slot_f1_at_05 - slot_null_f1_at_05``.
        slot_tp: Pooled slot true positives at ``threshold``.
        slot_fp: Pooled slot false positives at ``threshold``.
        slot_fn: Pooled slot false negatives at ``threshold``.
        micro_event_f1: Bipartite Hungarian ±20 ms event F1 of decoded slot centers.
        null_event_f1: Strongest audio-blind bipartite event-F1 floor.
        null_event_kind: Baseline kind achieving ``null_event_f1``.
        skill_event_f1: Headroom-normalized skill over ``null_event_f1``.
        micro_timing_match: Rank-indexed ``timing_match`` of decoded slot centers.
        n_pred: Pooled decoded onset count at ``threshold``.
        n_ref: Pooled ground-truth onset count.
        mean_abs_prob_delta: Mean ``|p_corrupt - p_none|`` across all beat slots.
        slot_flip_rate: Fraction of binary slot decisions changed vs ``none``.
        occupied_jaccard_distance: ``1 - IoU`` of occupied slots vs ``none``.
    """

    corruption: str
    slot_f1_at_05: float
    slot_f1_max: float
    slot_best_threshold: float
    slot_null_f1_at_05: float
    slot_skill_f1_at_05: float
    slot_tp: int
    slot_fp: int
    slot_fn: int
    micro_event_f1: float
    null_event_f1: float
    null_event_kind: str
    skill_event_f1: float
    micro_timing_match: float
    n_pred: int
    n_ref: int
    mean_abs_prob_delta: float
    slot_flip_rate: float
    occupied_jaccard_distance: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Serialize grounding condition metrics for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class AudioGroundingReport:
    """Audio-corruption grounding report across ``none``, ``shuffle``, ``zeros``.

    Attributes:
        threshold: Occupancy cutoff applied to slot probabilities.
        tolerance_sec: Time match window in seconds.
        n_charts: Charts evaluated.
        n_beats: Pooled integer beats across ``n_charts``.
        conditions: Mapping from corruption kind to its condition report.
        slot_f1_drop_shuffle: ``none.slot_f1_at_05 - shuffle.slot_f1_at_05``.
        slot_f1_drop_zeros: ``none.slot_f1_at_05 - zeros.slot_f1_at_05``.
        event_skill_drop_shuffle: ``none.skill_event_f1 - shuffle.skill_event_f1``.
        event_skill_drop_zeros: ``none.skill_event_f1 - zeros.skill_event_f1``.
    """

    threshold: float
    tolerance_sec: float
    n_charts: int
    n_beats: int
    conditions: dict[str, AudioGroundingConditionReport]
    slot_f1_drop_shuffle: float
    slot_f1_drop_zeros: float
    event_skill_drop_shuffle: float
    event_skill_drop_zeros: float

    def as_dict(self) -> dict[str, object]:
        """Serialize full audio-grounding report for JSON.

        Returns:
            JSON-serializable mapping.
        """
        return {
            "metric": "audio_grounding_ablation",
            "threshold": self.threshold,
            "tolerance_sec": self.tolerance_sec,
            "n_charts": self.n_charts,
            "n_beats": self.n_beats,
            "slot_f1_drop_shuffle": self.slot_f1_drop_shuffle,
            "slot_f1_drop_zeros": self.slot_f1_drop_zeros,
            "event_skill_drop_shuffle": self.event_skill_drop_shuffle,
            "event_skill_drop_zeros": self.event_skill_drop_zeros,
            "conditions": {
                name: cond.as_dict() for name, cond in self.conditions.items()
            },
        }


def _prediction_shift_stats(
    baseline_preds: list[np.ndarray],
    other_preds: list[np.ndarray],
    *,
    threshold: float,
) -> tuple[float, float, float]:
    """Compare corrupted slot probabilities and binary decisions against ``none``.

    Args:
        baseline_preds: Uncorrupted ``(n_beats, 48)`` probability arrays.
        other_preds: Corrupted probability arrays aligned with ``baseline_preds``.
        threshold: Occupancy cutoff for binary decisions.

    Returns:
        Tuple ``(mean_abs_prob_delta, slot_flip_rate, occupied_jaccard_distance)``.
    """
    base_all = np.concatenate(
        [np.asarray(arr, dtype=np.float64) for arr in baseline_preds],
        axis=0,
    )
    other_all = np.concatenate(
        [np.asarray(arr, dtype=np.float64) for arr in other_preds],
        axis=0,
    )
    mean_abs_delta = float(np.mean(np.abs(other_all - base_all)))
    base_bin = base_all >= float(threshold)
    other_bin = other_all >= float(threshold)
    flip_rate = float(np.mean(base_bin != other_bin))
    union = int(np.logical_or(base_bin, other_bin).sum())
    if union == 0:
        jaccard_dist = 0.0
    else:
        intersection = int(np.logical_and(base_bin, other_bin).sum())
        jaccard_dist = 1.0 - float(intersection) / float(union)
    return mean_abs_delta, flip_rate, jaccard_dist


def grounding_report_from_predictions(
    charts: list[datasets.DdclChart],
    predictions_by_corruption: dict[str, list[np.ndarray]],
    *,
    threshold: float = constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
    seed: int = onset_null_baseline.DEFAULT_SEED,
) -> AudioGroundingReport:
    """Score precomputed ``none`` / ``shuffle`` / ``zeros`` slot predictions.

    Args:
        charts: Loaded charts with BPM segments and ``gt_times_sec``.
        predictions_by_corruption: Mapping from each kind in :data:`CORRUPTION_KINDS`
            to per-chart ``(n_beats, 48)`` predicted probabilities.
        threshold: Occupancy cutoff for slot F1, decoded timing, and flip rates.
        thresholds: Sweep for ``slot_f1_max``.
        tolerance_sec: Match window in seconds.
        seed: Base RNG seed for beat-shuffle and audio-blind null baselines.

    Returns:
        Audio-grounding ablation report.

    Raises:
        ValueError: If ``charts`` is empty, BPM is missing, or predictions are missing.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    for chart in charts:
        if not chart.bpm_segments:
            raise ValueError(f"missing BPM segments for {chart.song_key}")
    for kind in CORRUPTION_KINDS:
        preds = predictions_by_corruption.get(kind)
        if preds is None or len(preds) != len(charts):
            raise ValueError(
                f"predictions_by_corruption[{kind!r}] must align with charts"
            )

    tgt_all = np.concatenate([chart.slots for chart in charts], axis=0)
    rng = np.random.default_rng(seed)
    null_pred = shuffle_slot_null(tgt_all, rng)
    slot_null_f1 = counts_at_threshold(null_pred, tgt_all, float(threshold)).f_score

    baseline_preds = predictions_by_corruption["none"]
    conditions: dict[str, AudioGroundingConditionReport] = {}
    for kind in CORRUPTION_KINDS:
        preds = predictions_by_corruption[kind]
        pred_all = np.concatenate(preds, axis=0)
        counts_primary = counts_at_threshold(pred_all, tgt_all, float(threshold))
        best_threshold = float(threshold)
        best_f1 = counts_primary.f_score
        for tau in thresholds:
            score = counts_at_threshold(pred_all, tgt_all, float(tau)).f_score
            if score > best_f1:
                best_f1 = score
                best_threshold = float(tau)

        masks = [
            (np.asarray(pred, dtype=np.float64) >= float(threshold)).astype(np.float64)
            for pred in preds
        ]
        timing_cond = _score_binary_masks(
            kind,
            charts,
            masks,
            tolerance_sec=tolerance_sec,
            seed=seed,
        )
        mean_abs_delta, flip_rate, jaccard_dist = _prediction_shift_stats(
            baseline_preds,
            preds,
            threshold=threshold,
        )
        conditions[kind] = AudioGroundingConditionReport(
            corruption=kind,
            slot_f1_at_05=counts_primary.f_score,
            slot_f1_max=best_f1,
            slot_best_threshold=best_threshold,
            slot_null_f1_at_05=slot_null_f1,
            slot_skill_f1_at_05=counts_primary.f_score - slot_null_f1,
            slot_tp=counts_primary.true_positives,
            slot_fp=counts_primary.false_positives,
            slot_fn=counts_primary.false_negatives,
            micro_event_f1=timing_cond.micro_event_f1,
            null_event_f1=timing_cond.null_event_f1,
            null_event_kind=timing_cond.null_event_kind,
            skill_event_f1=timing_cond.skill_event_f1,
            micro_timing_match=timing_cond.micro_timing_match,
            n_pred=timing_cond.n_pred,
            n_ref=timing_cond.n_ref,
            mean_abs_prob_delta=mean_abs_delta,
            slot_flip_rate=flip_rate,
            occupied_jaccard_distance=jaccard_dist,
        )

    none_cond = conditions["none"]
    shuffle_cond = conditions["shuffle"]
    zeros_cond = conditions["zeros"]
    n_beats = int(tgt_all.shape[0])
    return AudioGroundingReport(
        threshold=float(threshold),
        tolerance_sec=float(tolerance_sec),
        n_charts=len(charts),
        n_beats=n_beats,
        conditions=conditions,
        slot_f1_drop_shuffle=none_cond.slot_f1_at_05 - shuffle_cond.slot_f1_at_05,
        slot_f1_drop_zeros=none_cond.slot_f1_at_05 - zeros_cond.slot_f1_at_05,
        event_skill_drop_shuffle=(
            none_cond.skill_event_f1 - shuffle_cond.skill_event_f1
        ),
        event_skill_drop_zeros=none_cond.skill_event_f1 - zeros_cond.skill_event_f1,
    )


def evaluate_audio_grounding(
    model: SlotPredictor,
    charts: list[datasets.DdclChart],
    *,
    threshold: float = constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
    seed: int = onset_null_baseline.DEFAULT_SEED,
) -> AudioGroundingReport:
    """Run ``none``, ``shuffle``, and ``zeros`` audio-corruption grounding checks.

    Corrupts ``beat_audio`` while leaving ``stream`` (``[meter, bpm]``) intact to
    measure how much slot and decoded ±20 ms skill depends on audio rather than
    metric templates.

    Args:
        model: Keras model with DDCL placement inputs.
        charts: Loaded beat-grid charts with BPM and onset times.
        threshold: Occupancy cutoff for slot F1, decoded timing, and flip rates.
        thresholds: Sweep for ``slot_f1_max``.
        tolerance_sec: Match window in seconds.
        seed: Base RNG seed for shuffling and null baselines.

    Returns:
        Audio-grounding ablation report across all corruption modes.
    """
    predictions_by_corruption: dict[str, list[np.ndarray]] = {}
    for kind in CORRUPTION_KINDS:
        corrupted_preds = [
            predict_chart_slots(
                model,
                corrupt_chart_audio(chart, kind, seed=seed + idx),
            )
            for idx, chart in enumerate(charts)
        ]
        predictions_by_corruption[kind] = corrupted_preds
    return grounding_report_from_predictions(
        charts,
        predictions_by_corruption,
        threshold=threshold,
        thresholds=thresholds,
        tolerance_sec=tolerance_sec,
        seed=seed,
    )
