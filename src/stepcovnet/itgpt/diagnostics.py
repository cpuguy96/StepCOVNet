"""Structured error-breakdown diagnostics for ITGPT 48-slot onset predictions."""

from __future__ import annotations

import collections
import dataclasses
import typing

import numpy as np
import torch
from torch import nn

from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import constants, evaluation

DEFAULT_CALIBRATION_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.10),
    (0.10, 0.25),
    (0.25, 0.40),
    (0.40, 0.50),
    (0.50, 0.65),
    (0.65, 0.80),
    (0.80, 0.95),
    (0.95, 1.01),
)
DEFAULT_WORST_K = 8
DEFAULT_BEST_K = 5
EXACT_METER_MAX_DISTINCT = 12


def _subdivision_groups() -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return disjoint 48-slot index groups ordered by rhythmic resolution.

    Returns:
        Tuple of ``(group_name, slot_indices)`` pairs partitioning ``0..47``.
    """
    set_quarter = {0}
    set_8th = {24}
    set_16th_off = {12, 36}
    set_24th = set(constants.INDICES_24TH) - set_quarter - set_8th - set_16th_off
    set_32nd = (
        set(constants.INDICES_32ND) - set_quarter - set_8th - set_16th_off - set_24th
    )
    set_micro = (
        set(range(constants.N_SLOTS))
        - set_quarter
        - set_8th
        - set_16th_off
        - set_24th
        - set_32nd
    )
    return (
        ("quarter", tuple(sorted(set_quarter))),
        ("8th", tuple(sorted(set_8th))),
        ("16th_off", tuple(sorted(set_16th_off))),
        ("24th", tuple(sorted(set_24th))),
        ("32nd", tuple(sorted(set_32nd))),
        ("micro", tuple(sorted(set_micro))),
    )


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 with zero-safe denominators.

    Args:
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.

    Returns:
        Tuple ``(precision, recall, f1)``.
    """
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    return prec, rec, f1


@dataclasses.dataclass(frozen=True)
class SliceCounts:
    """Binary slot detection metrics for one slice of the evaluation set.

    Attributes:
        name: Human-readable slice label.
        tp: True positive slot count.
        fp: False positive slot count.
        fn: False negative slot count.
        gt: Ground-truth positive slot count (``tp + fn``).
        pred: Predicted positive slot count (``tp + fp``).
        precision: Precision ``tp / (tp + fp)``.
        recall: Recall ``tp / (tp + fn)``.
        f1: Binary F1 score.
        pct_errors: Percentage of global ``(fp + fn)`` contributed by this slice.
        n_charts: Optional chart count for chart-grouped slices (pack, meter).
    """

    name: str
    tp: int
    fp: int
    fn: int
    gt: int
    pred: int
    precision: float
    recall: float
    f1: float
    pct_errors: float
    n_charts: int | None = None

    def as_dict(self) -> dict[str, object]:
        """Serialize slice counts to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of this slice.
        """
        payload: dict[str, object] = {
            "name": self.name,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "gt": self.gt,
            "pred": self.pred,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "pct_errors": self.pct_errors,
        }
        if self.n_charts is not None:
            payload["n_charts"] = self.n_charts
        return payload


def _make_slice(
    name: str,
    tp: int,
    fp: int,
    fn: int,
    *,
    total_errors: int,
    n_charts: int | None = None,
) -> SliceCounts:
    """Build a :class:`SliceCounts` from raw binary confusion counts.

    Args:
        name: Slice label.
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.
        total_errors: Global ``fp + fn`` across the full evaluation set.
        n_charts: Optional number of charts in this slice.

    Returns:
        Populated :class:`SliceCounts`.
    """
    prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
    pct_err = (100.0 * (fp + fn) / total_errors) if total_errors else 0.0
    return SliceCounts(
        name=name,
        tp=tp,
        fp=fp,
        fn=fn,
        gt=tp + fn,
        pred=tp + fp,
        precision=prec,
        recall=rec,
        f1=f1,
        pct_errors=pct_err,
        n_charts=n_charts,
    )


@dataclasses.dataclass(frozen=True)
class TimingOffsetDistribution:
    """Nearest-opposite-event distance buckets on the flat 48-slot timeline.

    Attributes:
        total: Total error events classified (sum of all six buckets).
        le_1_slot: Errors within 1 slot (micro localization jitter).
        slots_2_3: Errors 2 to 3 slots away (``<= 1/16`` beat).
        slots_4_6: Errors 4 to 6 slots away (``<= 1/8`` beat).
        slots_7_12: Errors 7 to 12 slots away (``<= 1/4`` beat).
        other_same_beat: Errors ``> 12`` slots away on a beat with an opposite event.
        other_beat: Errors on a beat containing no opposite event (or empty opposite).
    """

    total: int
    le_1_slot: int
    slots_2_3: int
    slots_4_6: int
    slots_7_12: int
    other_same_beat: int
    other_beat: int

    def _pct(self, count: int) -> float:
        """Return percentage share of ``count`` relative to ``total``.

        Args:
            count: Bucket count.

        Returns:
            Percentage in ``[0.0, 100.0]``.
        """
        return (100.0 * count / self.total) if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        """Serialize counts and percentages to a JSON-compatible dictionary.

        Returns:
            Dictionary with ``total``, ``counts``, and ``pct`` mappings.
        """
        return {
            "total": self.total,
            "counts": {
                "le_1_slot": self.le_1_slot,
                "slots_2_3": self.slots_2_3,
                "slots_4_6": self.slots_4_6,
                "slots_7_12": self.slots_7_12,
                "other_same_beat": self.other_same_beat,
                "other_beat": self.other_beat,
            },
            "pct": {
                "le_1_slot": self._pct(self.le_1_slot),
                "slots_2_3": self._pct(self.slots_2_3),
                "slots_4_6": self._pct(self.slots_4_6),
                "slots_7_12": self._pct(self.slots_7_12),
                "other_same_beat": self._pct(self.other_same_beat),
                "other_beat": self._pct(self.other_beat),
            },
        }


@dataclasses.dataclass(frozen=True)
class CalibrationBin:
    """Empirical positive rate inside one predicted probability interval.

    Attributes:
        low: Inclusive lower probability bound.
        high: Exclusive upper probability bound.
        count: Number of slots falling in ``[low, high)``.
        mean_prob: Mean predicted probability in the bin.
        empirical_pos_rate: Fraction of ground-truth positive slots in the bin.
    """

    low: float
    high: float
    count: int
    mean_prob: float
    empirical_pos_rate: float

    def as_dict(self) -> dict[str, float | int]:
        """Serialize calibration bin statistics.

        Returns:
            Dictionary with bin bounds, count, mean probability, and positive rate.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ChartErrorRow:
    """Per-chart summary row for worst/best chart inspection.

    Attributes:
        song_key: Chart song identifier (``pack/song``).
        difficulty: StepMania difficulty label.
        meter: Numeric chart meter rating.
        n_beats: Evaluated beat count.
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        precision: Chart precision.
        recall: Chart recall.
        f1: Chart F1 score.
    """

    song_key: str
    difficulty: str
    meter: int
    n_beats: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, object]:
        """Serialize chart row to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of this chart row.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ErrorBreakdownReport:
    """Comprehensive diagnostic breakdown of ITGPT 48-slot onset errors.

    Attributes:
        threshold: Probability threshold used for binary slot decisions.
        n_charts: Total charts evaluated.
        n_beats: Total integer beats evaluated.
        overall: Global pooled binary slot metrics.
        total_errors: Global ``fp + fn`` count.
        subdivisions: Metrics partitioned by rhythmic subdivision.
        fp_timing_offsets: Distance distribution from false positives to nearest GT.
        fn_timing_offsets: Distance distribution from false negatives to nearest pred.
        by_pack: Metrics grouped by song pack prefix.
        by_meter: Metrics grouped by chart meter or meter bucket.
        by_beat_in_measure: Metrics grouped by ``beat % 4`` phase.
        by_song_section: Metrics grouped by ``intro``, ``body``, and ``outro``.
        calibration: Predicted probability vs empirical positive rate bins.
        worst_charts: Lowest-F1 charts ordered worst first.
        best_charts: Highest-F1 charts ordered best first.
    """

    threshold: float
    n_charts: int
    n_beats: int
    overall: SliceCounts
    total_errors: int
    subdivisions: tuple[SliceCounts, ...]
    fp_timing_offsets: TimingOffsetDistribution
    fn_timing_offsets: TimingOffsetDistribution
    by_pack: tuple[SliceCounts, ...]
    by_meter: tuple[SliceCounts, ...]
    by_beat_in_measure: tuple[SliceCounts, ...]
    by_song_section: tuple[SliceCounts, ...]
    calibration: tuple[CalibrationBin, ...]
    worst_charts: tuple[ChartErrorRow, ...]
    best_charts: tuple[ChartErrorRow, ...]

    def as_dict(self) -> dict[str, object]:
        """Serialize the full breakdown report for JSON output.

        Returns:
            JSON-serializable mapping with both structured lists and quick lookups.
        """
        return {
            "metric": "itgpt_error_breakdown",
            "threshold": self.threshold,
            "n_charts": self.n_charts,
            "n_beats": self.n_beats,
            "total_errors": self.total_errors,
            "overall": self.overall.as_dict(),
            "subdivision_f1": {item.name: item.f1 for item in self.subdivisions},
            "section_f1": {item.name: item.f1 for item in self.by_song_section},
            "subdivisions": [item.as_dict() for item in self.subdivisions],
            "fp_timing_offsets": self.fp_timing_offsets.as_dict(),
            "fn_timing_offsets": self.fn_timing_offsets.as_dict(),
            "by_pack": [item.as_dict() for item in self.by_pack],
            "by_meter": [item.as_dict() for item in self.by_meter],
            "by_beat_in_measure": [item.as_dict() for item in self.by_beat_in_measure],
            "by_song_section": [item.as_dict() for item in self.by_song_section],
            "calibration": [item.as_dict() for item in self.calibration],
            "worst_charts": [item.as_dict() for item in self.worst_charts],
            "best_charts": [item.as_dict() for item in self.best_charts],
        }

    def format_text(self) -> str:
        """Format a compact human-readable diagnostic table.

        Returns:
            Multi-line summary string suitable for stdout or research logs.
        """
        lines = [
            (
                f"OVERALL (tau={self.threshold:.2f}, charts={self.n_charts}, "
                f"beats={self.n_beats}): F1={self.overall.f1:.4f} "
                f"(P={self.overall.precision:.4f}, R={self.overall.recall:.4f}) | "
                f"TP={self.overall.tp} FP={self.overall.fp} FN={self.overall.fn} | "
                f"err={self.total_errors}"
            ),
            "",
            "--- 1. BY RHYTHMIC SUBDIVISION ---",
            (
                f"{'Group':<12} {'GT':>7} {'Pred':>7} {'TP':>7} {'FP':>6} {'FN':>6} "
                f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'%Err':>6}"
            ),
        ]
        for item in self.subdivisions:
            lines.append(
                f"{item.name:<12} {item.gt:>7} {item.pred:>7} {item.tp:>7} "
                f"{item.fp:>6} {item.fn:>6} {item.precision:>6.3f} "
                f"{item.recall:>6.3f} {item.f1:>6.3f} {item.pct_errors:>5.1f}%"
            )

        lines.extend(
            [
                "",
                "--- 2. FP & FN TIMING OFFSET ANALYSIS ---",
                _format_offset_line("FP", self.fp_timing_offsets),
                _format_offset_line("FN", self.fn_timing_offsets),
                "",
                "--- 3. BY PACK ---",
            ]
        )
        for item in self.by_pack:
            lines.append(
                f"  {item.name:<35} charts={item.n_charts or 0:>3} "
                f"F1={item.f1:.4f} (P={item.precision:.3f} R={item.recall:.3f}) "
                f"FP={item.fp:>5} FN={item.fn:>5}"
            )

        lines.extend(["", "--- 4. BY METER ---"])
        for item in self.by_meter:
            lines.append(
                f"  {item.name:<12} charts={item.n_charts or 0:>3} "
                f"F1={item.f1:.4f} (P={item.precision:.3f} R={item.recall:.3f}) "
                f"FP={item.fp:>5} FN={item.fn:>5}"
            )

        lines.extend(["", "--- 5. BEAT PHASE & SONG SECTION ---"])
        for item in self.by_beat_in_measure:
            lines.append(
                f"  {item.name:<14} F1={item.f1:.4f} "
                f"(P={item.precision:.3f} R={item.recall:.3f}) "
                f"TP={item.tp:>5} FP={item.fp:>4} FN={item.fn:>4}"
            )
        for item in self.by_song_section:
            lines.append(
                f"  {item.name:<14} F1={item.f1:.4f} "
                f"(P={item.precision:.3f} R={item.recall:.3f}) "
                f"TP={item.tp:>5} FP={item.fp:>4} FN={item.fn:>4}"
            )

        lines.extend(["", "--- 6. PROBABILITY CALIBRATION ---"])
        for cbin in self.calibration:
            lines.append(
                f"  [{cbin.low:.2f}, {cbin.high:.2f}): count={cbin.count:>7} "
                f"mean_p={cbin.mean_prob:.3f} emp_pos={cbin.empirical_pos_rate:.3f}"
            )

        lines.extend(["", f"--- 7. WORST {len(self.worst_charts)} CHARTS ---"])
        for row in self.worst_charts:
            lines.append(
                f"  F1={row.f1:.3f} (P={row.precision:.2f} R={row.recall:.2f} "
                f"FP={row.fp:>4} FN={row.fn:>4}) m={row.meter:>2} "
                f"{row.difficulty:<9} {row.song_key}"
            )
        lines.append(f"--- BEST {len(self.best_charts)} CHARTS ---")
        for row in self.best_charts:
            lines.append(
                f"  F1={row.f1:.3f} (P={row.precision:.2f} R={row.recall:.2f} "
                f"FP={row.fp:>4} FN={row.fn:>4}) m={row.meter:>2} "
                f"{row.difficulty:<9} {row.song_key}"
            )
        return "\n".join(lines)


def _format_offset_line(label: str, dist: TimingOffsetDistribution) -> str:
    """Format one timing-offset summary line.

    Args:
        label: Prefix label (``"FP"`` or ``"FN"``).
        dist: Timing-offset counts to render.

    Returns:
        Formatted single-line percentage breakdown.
    """
    pct = typing.cast(dict[str, float], dist.as_dict()["pct"])
    return (
        f"{label} ({dist.total}): <=1 slot: {pct['le_1_slot']:.1f}% | "
        f"2-3 slots: {pct['slots_2_3']:.1f}% | "
        f"4-6 slots: {pct['slots_4_6']:.1f}% | "
        f"7-12 slots: {pct['slots_7_12']:.1f}% | "
        f"other same-beat: {pct['other_same_beat']:.1f}% | "
        f"other beat: {pct['other_beat']:.1f}%"
    )


def _accumulate_offsets(
    err_pos: np.ndarray,
    ref_pos: np.ndarray,
    counts: list[int],
) -> None:
    """Add nearest-reference distance bucket counts for one chart in-place.

    Args:
        err_pos: Sorted flat 48-slot indices of error events (FP or FN).
        ref_pos: Sorted flat 48-slot indices of opposite events (GT or pred).
        counts: Mutable 6-element accumulator list corresponding to
            ``[<=1, 2-3, 4-6, 7-12, other_same_beat, other_beat]``.
    """
    if err_pos.size == 0:
        return
    if ref_pos.size == 0:
        counts[5] += int(err_pos.size)
        return
    idx = np.searchsorted(ref_pos, err_pos)
    right_idx = np.minimum(idx, ref_pos.size - 1)
    left_idx = np.maximum(idx - 1, 0)
    d_right = np.where(idx < ref_pos.size, np.abs(ref_pos[right_idx] - err_pos), 10**9)
    d_left = np.where(idx > 0, np.abs(err_pos - ref_pos[left_idx]), 10**9)
    dist = np.minimum(d_left, d_right)

    counts[0] += int((dist <= 1).sum())
    counts[1] += int(((dist > 1) & (dist <= 3)).sum())
    counts[2] += int(((dist > 3) & (dist <= 6)).sum())
    counts[3] += int(((dist > 6) & (dist <= 12)).sum())
    far_mask = dist > 12
    if np.any(far_mask):
        ref_beats = set((ref_pos // constants.N_SLOTS).tolist())
        far_beats = (err_pos[far_mask] // constants.N_SLOTS).tolist()
        same_beat = sum(1 for beat in far_beats if beat in ref_beats)
        counts[4] += same_beat
        counts[5] += len(far_beats) - same_beat


def _meter_label(meter: int, *, exact_meter: bool) -> str:
    """Map numeric meter to an exact or bucketed display label.

    Args:
        meter: StepMania chart difficulty meter.
        exact_meter: If True, format exact meter (``meter_09``); else bucket.

    Returns:
        Sortable meter label string.
    """
    if exact_meter:
        return f"meter_{meter:02d}"
    if meter <= 8:
        return "01_<=8"
    if meter <= 10:
        return "02_9-10"
    if meter <= 12:
        return "03_11-12"
    return "04_13+"


def error_breakdown_from_predictions(
    charts: typing.Sequence[ddcl_datasets.DdclChart],
    predictions: typing.Sequence[np.ndarray],
    *,
    threshold: float = constants.THRESHOLD_05,
    exact_meter: bool | None = None,
    worst_k: int = DEFAULT_WORST_K,
    best_k: int = DEFAULT_BEST_K,
) -> ErrorBreakdownReport:
    """Compute full diagnostic error breakdown from precomputed slot predictions.

    Args:
        charts: Evaluated charts with ground-truth ``slots``.
        predictions: Per-chart slot probability arrays aligned with ``charts``.
        threshold: Probability cutoff for binary slot decisions.
        exact_meter: Whether to group by exact meter rather than coarse buckets.
            Defaults to exact grouping when at most 12 distinct meters appear.
        worst_k: Maximum number of lowest-F1 charts to retain.
        best_k: Maximum number of highest-F1 charts to retain.

    Returns:
        Structured :class:`ErrorBreakdownReport`.

    Raises:
        ValueError: If ``charts`` is empty or ``len(charts) != len(predictions)``.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    if len(charts) != len(predictions):
        raise ValueError(
            f"charts length ({len(charts)}) != predictions length ({len(predictions)})"
        )

    aligned_preds: list[np.ndarray] = []
    aligned_tgts: list[np.ndarray] = []
    for chart, raw_pred in zip(charts, predictions, strict=True):
        pred = np.asarray(raw_pred, dtype=np.float32)
        n_beats = min(int(pred.shape[0]), int(chart.slots.shape[0]))
        aligned_preds.append(pred[:n_beats])
        aligned_tgts.append(np.asarray(chart.slots[:n_beats], dtype=np.float32))

    pred_all = np.concatenate(aligned_preds, axis=0)
    tgt_all = np.concatenate(aligned_tgts, axis=0)
    pbin_all = pred_all >= float(threshold)
    tbin_all = tgt_all >= 0.5

    tot_tp = int(np.logical_and(pbin_all, tbin_all).sum())
    tot_fp = int(np.logical_and(pbin_all, ~tbin_all).sum())
    tot_fn = int(np.logical_and(~pbin_all, tbin_all).sum())
    tot_err = tot_fp + tot_fn
    overall = _make_slice(
        "overall",
        tot_tp,
        tot_fp,
        tot_fn,
        total_errors=tot_err,
        n_charts=len(charts),
    )

    subdivisions: list[SliceCounts] = []
    for group_name, idxs in _subdivision_groups():
        pb = pbin_all[:, idxs]
        tb = tbin_all[:, idxs]
        subdivisions.append(
            _make_slice(
                group_name,
                int(np.logical_and(pb, tb).sum()),
                int(np.logical_and(pb, ~tb).sum()),
                int(np.logical_and(~pb, tb).sum()),
                total_errors=tot_err,
            )
        )

    fp_buckets = [0, 0, 0, 0, 0, 0]
    fn_buckets = [0, 0, 0, 0, 0, 0]
    use_exact_meter = (
        len({chart.meter for chart in charts}) <= EXACT_METER_MAX_DISTINCT
        if exact_meter is None
        else exact_meter
    )
    pack_acc: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    meter_acc: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    phase_acc = [[0, 0, 0] for _ in range(4)]
    section_names = ("intro", "body", "outro")
    section_acc: dict[str, list[int]] = {name: [0, 0, 0] for name in section_names}
    chart_rows: list[ChartErrorRow] = []

    for chart, pred, tgt in zip(charts, aligned_preds, aligned_tgts, strict=True):
        pb = pred >= float(threshold)
        tb = tgt >= 0.5
        n_beats = int(pb.shape[0])

        pb_flat = pb.reshape(-1)
        tb_flat = tb.reshape(-1)
        _accumulate_offsets(
            np.flatnonzero(pb_flat & ~tb_flat),
            np.flatnonzero(tb_flat),
            fp_buckets,
        )
        _accumulate_offsets(
            np.flatnonzero(~pb_flat & tb_flat),
            np.flatnonzero(pb_flat),
            fn_buckets,
        )

        tp = int(np.logical_and(pb, tb).sum())
        fp = int(np.logical_and(pb, ~tb).sum())
        fn = int(np.logical_and(~pb, tb).sum())
        prec, rec, f1 = _precision_recall_f1(tp, fp, fn)
        chart_rows.append(
            ChartErrorRow(
                song_key=chart.song_key,
                difficulty=chart.difficulty,
                meter=chart.meter,
                n_beats=n_beats,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=prec,
                recall=rec,
                f1=f1,
            )
        )

        pack = chart.song_key.split("/", maxsplit=1)[0]
        pack_acc[pack][0] += tp
        pack_acc[pack][1] += fp
        pack_acc[pack][2] += fn
        pack_acc[pack][3] += 1

        mlabel = _meter_label(chart.meter, exact_meter=use_exact_meter)
        meter_acc[mlabel][0] += tp
        meter_acc[mlabel][1] += fp
        meter_acc[mlabel][2] += fn
        meter_acc[mlabel][3] += 1

        for beat_mod in range(4):
            sl = slice(beat_mod, n_beats, 4)
            phase_acc[beat_mod][0] += int(np.logical_and(pb[sl], tb[sl]).sum())
            phase_acc[beat_mod][1] += int(np.logical_and(pb[sl], ~tb[sl]).sum())
            phase_acc[beat_mod][2] += int(np.logical_and(~pb[sl], tb[sl]).sum())

        i1 = min(n_beats, max(1, int(0.1 * n_beats))) if n_beats else 0
        i2 = min(n_beats, max(i1, int(0.9 * n_beats))) if n_beats else 0
        for sec_name, sl in (
            ("intro", slice(0, i1)),
            ("body", slice(i1, i2)),
            ("outro", slice(i2, n_beats)),
        ):
            section_acc[sec_name][0] += int(np.logical_and(pb[sl], tb[sl]).sum())
            section_acc[sec_name][1] += int(np.logical_and(pb[sl], ~tb[sl]).sum())
            section_acc[sec_name][2] += int(np.logical_and(~pb[sl], tb[sl]).sum())

    fp_dist = TimingOffsetDistribution(
        total=sum(fp_buckets),
        le_1_slot=fp_buckets[0],
        slots_2_3=fp_buckets[1],
        slots_4_6=fp_buckets[2],
        slots_7_12=fp_buckets[3],
        other_same_beat=fp_buckets[4],
        other_beat=fp_buckets[5],
    )
    fn_dist = TimingOffsetDistribution(
        total=sum(fn_buckets),
        le_1_slot=fn_buckets[0],
        slots_2_3=fn_buckets[1],
        slots_4_6=fn_buckets[2],
        slots_7_12=fn_buckets[3],
        other_same_beat=fn_buckets[4],
        other_beat=fn_buckets[5],
    )

    by_pack = tuple(
        _make_slice(pack, tp, fp, fn, total_errors=tot_err, n_charts=n_ch)
        for pack, (tp, fp, fn, n_ch) in sorted(pack_acc.items())
    )
    by_meter = tuple(
        _make_slice(mlabel, tp, fp, fn, total_errors=tot_err, n_charts=n_ch)
        for mlabel, (tp, fp, fn, n_ch) in sorted(meter_acc.items())
    )
    by_beat_in_measure = tuple(
        _make_slice(f"beat_mod4_{idx}", tp, fp, fn, total_errors=tot_err)
        for idx, (tp, fp, fn) in enumerate(phase_acc)
    )
    by_song_section = tuple(
        _make_slice(
            name,
            section_acc[name][0],
            section_acc[name][1],
            section_acc[name][2],
            total_errors=tot_err,
        )
        for name in section_names
    )

    p_flat = pred_all.reshape(-1)
    t_flat = tbin_all.reshape(-1)
    calibration_bins: list[CalibrationBin] = []
    for low, high in DEFAULT_CALIBRATION_BINS:
        mask = (p_flat >= low) & (p_flat < high)
        count = int(mask.sum())
        if count > 0:
            calibration_bins.append(
                CalibrationBin(
                    low=low,
                    high=high,
                    count=count,
                    mean_prob=float(p_flat[mask].mean()),
                    empirical_pos_rate=float(t_flat[mask].mean()),
                )
            )

    ordered_rows = sorted(chart_rows, key=lambda r: (r.f1, r.song_key, r.difficulty))
    worst_charts = tuple(ordered_rows[: max(0, worst_k)])
    best_charts = tuple(reversed(ordered_rows[-max(0, best_k) :])) if best_k > 0 else ()

    return ErrorBreakdownReport(
        threshold=float(threshold),
        n_charts=len(charts),
        n_beats=int(tgt_all.shape[0]),
        overall=overall,
        total_errors=tot_err,
        subdivisions=tuple(subdivisions),
        fp_timing_offsets=fp_dist,
        fn_timing_offsets=fn_dist,
        by_pack=by_pack,
        by_meter=by_meter,
        by_beat_in_measure=by_beat_in_measure,
        by_song_section=by_song_section,
        calibration=tuple(calibration_bins),
        worst_charts=worst_charts,
        best_charts=best_charts,
    )


def evaluate_error_breakdown(
    model: nn.Module | evaluation.SlotPredictor,
    charts: typing.Sequence[ddcl_datasets.DdclChart],
    *,
    threshold: float = constants.THRESHOLD_05,
    max_beats: int = constants.MAX_BEATS,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
    exact_meter: bool | None = None,
    worst_k: int = DEFAULT_WORST_K,
    best_k: int = DEFAULT_BEST_K,
) -> ErrorBreakdownReport:
    """Predict slot probabilities and compute the diagnostic error breakdown.

    Args:
        model: Trained PyTorch ITGPT placement model or :class:`SlotPredictor`.
        charts: Loaded beat-grid charts to analyze.
        threshold: Probability cutoff for binary slot decisions.
        max_beats: Pad cap used at train time.
        device: Optional inference device.
        amp_dtype: Optional inference autocast dtype.
        exact_meter: Optional override for exact vs bucketed meter slices.
        worst_k: Maximum number of lowest-F1 charts to retain.
        best_k: Maximum number of highest-F1 charts to retain.

    Returns:
        Structured :class:`ErrorBreakdownReport`.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    predictions = [
        evaluation.predict_chart_slots(
            model,
            chart,
            max_beats=max_beats,
            device=device,
            amp_dtype=amp_dtype,
        )
        for chart in charts
    ]
    return error_breakdown_from_predictions(
        charts,
        predictions,
        threshold=threshold,
        exact_meter=exact_meter,
        worst_k=worst_k,
        best_k=best_k,
    )
