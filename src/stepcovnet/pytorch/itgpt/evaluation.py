"""``M-slot48`` evaluation and audio-grounding for PyTorch ITGPT."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
import torch
from torch import nn

from stepcovnet import timing_match
from stepcovnet.ddc import peak_pick
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.ddcl import slots as ddcl_slots
from stepcovnet.itgpt import constants
from stepcovnet.pytorch import runtime
from stepcovnet.pytorch.itgpt import datasets

DEFAULT_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
CORRUPTION_KINDS = ("none", "shuffle", "zeros")
DEFAULT_GROUNDING_SEED = 0


class SlotPredictor(typing.Protocol):
    """Minimal predict interface used by ``M-slot48`` eval."""

    def predict(self, inputs: object, verbose: int = 0) -> object:
        """Return slot probabilities.

        Args:
            inputs: Dict of NumPy arrays.
            verbose: Compatibility verbosity flag.

        Returns:
            Array-like slot scores.
        """


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
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> Slot48Counts:
    """Score one ``(n_beats, 48)`` pair at a decision threshold.

    Args:
        pred: Predicted slot probabilities.
        target: Ground-truth 0/1 slots.
        threshold: Decision threshold.

    Returns:
        Pooled :class:`Slot48Counts`.
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
        rng: NumPy RNG.

    Returns:
        Beat-shuffled copy of ``target``.
    """
    out = np.asarray(target, dtype=np.float32).copy()
    order = rng.permutation(out.shape[0])
    return out[order]


@dataclasses.dataclass(frozen=True)
class Slot48EvalReport:
    """Pooled ``M-slot48`` scores matching :class:`stepcovnet.ddcl.evaluation.Slot48EvalReport`.

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

    def as_dict(self) -> dict[str, object]:
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
            "citation": "omalley2026itgpt",
        }


def corrupt_chart_audio(
    chart: ddcl_datasets.DdclChart,
    corruption: str,
    *,
    seed: int = DEFAULT_GROUNDING_SEED,
) -> ddcl_datasets.DdclChart:
    """Return a chart copy with ``beat_audio`` corrupted and metadata intact.

    Args:
        chart: Source beat-grid chart.
        corruption: One of ``"none"``, ``"shuffle"``, or ``"zeros"``.
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
        micro_event_f1: ±20 ms event F1 of decoded slot centers vs GT onsets.
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
        event_f1_drop_shuffle: ``none.micro_event_f1 - shuffle.micro_event_f1``.
        event_f1_drop_zeros: ``none.micro_event_f1 - zeros.micro_event_f1``.
    """

    threshold: float
    tolerance_sec: float
    n_charts: int
    n_beats: int
    conditions: dict[str, AudioGroundingConditionReport]
    slot_f1_drop_shuffle: float
    slot_f1_drop_zeros: float
    event_f1_drop_shuffle: float
    event_f1_drop_zeros: float

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
            "event_f1_drop_shuffle": self.event_f1_drop_shuffle,
            "event_f1_drop_zeros": self.event_f1_drop_zeros,
            "conditions": {
                name: cond.as_dict() for name, cond in self.conditions.items()
            },
        }


class TorchSlotPredictor:
    """Adapter exposing a PyTorch model via :class:`SlotPredictor`.

    Args:
        model: PyTorch ITGPT placement model returning slot probabilities.
        device: Target compute device. Defaults to the model's parameter device.
        amp_dtype: Optional autocast dtype during inference (``None`` for fp32).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | None = None,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        if device is None:
            first_param = next(model.parameters(), None)
            self.device = (
                first_param.device if first_param is not None else torch.device("cpu")
            )
        else:
            self.device = device
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def predict(self, inputs: object, verbose: int = 0) -> np.ndarray:
        """Run inference in eval mode and return float32 NumPy probabilities.

        Args:
            inputs: Dictionary of NumPy arrays (``audio``, ``bpm``, ``difficulty``).
            verbose: Unused compatibility verbosity parameter.

        Returns:
            Float32 NumPy array of shape ``(batch, beats, 48)``.

        Raises:
            TypeError: If ``inputs`` is not a dict of array-like objects.
        """
        del verbose
        if not isinstance(inputs, dict):
            raise TypeError(
                f"Expected dict of NumPy inputs, got {type(inputs).__name__}"
            )
        was_training = self.model.training
        self.model.eval()
        try:
            tensor_inputs = {
                str(key): torch.as_tensor(val, dtype=torch.float32, device=self.device)
                for key, val in inputs.items()
            }
            with runtime.autocast_context(self.device, self.amp_dtype):
                probs = self.model(tensor_inputs)
            return probs.detach().float().cpu().numpy()
        finally:
            if was_training:
                self.model.train()


def _as_predictor(
    model: nn.Module | SlotPredictor,
    *,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> SlotPredictor:
    """Wrap a PyTorch ``nn.Module`` in :class:`TorchSlotPredictor` if needed.

    Args:
        model: PyTorch model or an existing :class:`SlotPredictor`.
        device: Optional inference device.
        amp_dtype: Optional inference autocast dtype.

    Returns:
        Object satisfying :class:`SlotPredictor`.
    """
    if isinstance(model, nn.Module):
        return TorchSlotPredictor(model, device=device, amp_dtype=amp_dtype)
    return model


def _difficulty_source(model: nn.Module | SlotPredictor) -> str:
    """Resolve ``difficulty_source`` from a model or predictor wrapper.

    Args:
        model: PyTorch model or :class:`SlotPredictor`.

    Returns:
        Configured ``difficulty_source`` (defaults to ``"meter"``).
    """
    unwrapped = getattr(model, "model", model)
    unwrapped = getattr(unwrapped, "_orig_mod", unwrapped)
    return str(
        getattr(unwrapped, "difficulty_source", constants.DIFFICULTY_SOURCE_METER)
    )


def predict_chart_slots(
    model: nn.Module | SlotPredictor,
    chart: ddcl_datasets.DdclChart,
    *,
    max_beats: int | None = None,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Return slot probabilities for one chart (true beats only).

    Args:
        model: PyTorch ITGPT placement model or :class:`SlotPredictor`.
        chart: Loaded beat-grid chart.
        max_beats: Optional pad cap used at train time. Defaults to the
            chart's chunk-aligned length.
        device: Optional inference device.
        amp_dtype: Optional autocast dtype (default fp32 for eval parity).

    Returns:
        Probabilities ``(n_beats, 48)``.
    """
    predictor = _as_predictor(model, device=device, amp_dtype=amp_dtype)
    diff_source = _difficulty_source(predictor)
    cap = (
        max(datasets.pad_length(chart.n_beats, chart.n_beats), constants.CHUNK_ALIGN)
        if max_beats is None
        else max(int(max_beats), constants.CHUNK_ALIGN)
    )
    inputs, _, mask = datasets.pack_chart(
        chart, max_beats=cap, difficulty_source=diff_source
    )
    pred = np.asarray(predictor.predict(inputs, verbose=0), dtype=np.float32)
    if pred.ndim == 2:
        pred = pred[None, ...]
    n_beats = int(mask[0].sum())
    return pred[0, :n_beats]


def evaluate_slot48(
    model: nn.Module | SlotPredictor,
    charts: list[ddcl_datasets.DdclChart],
    *,
    seed: int = 42,
    max_beats: int = constants.MAX_BEATS,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> Slot48EvalReport:
    """Pool ``M-slot48`` over full-chart ITGPT predictions.

    Args:
        model: Trained PyTorch placement model or :class:`SlotPredictor`.
        charts: Loaded val charts.
        seed: Null-shuffle seed.
        max_beats: Pad cap used at train time.
        device: Optional inference device.
        amp_dtype: Optional autocast dtype (default ``None`` / fp32).

    Returns:
        Pooled :class:`Slot48EvalReport`.

    Raises:
        ValueError: If ``charts`` is empty.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    predictor = _as_predictor(model, device=device, amp_dtype=amp_dtype)
    diff_source = _difficulty_source(predictor)
    rng = np.random.default_rng(seed)
    pred_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    for chart in charts:
        inputs, _, mask = datasets.pack_chart(
            chart, max_beats=max_beats, difficulty_source=diff_source
        )
        pred = np.asarray(predictor.predict(inputs, verbose=0), dtype=np.float32)
        if pred.ndim == 2:
            pred = pred[None, ...]
        n_beats = int(mask[0].sum())
        pred_parts.append(pred[0, :n_beats])
        tgt_parts.append(chart.slots[:n_beats])
    pred_all = np.concatenate(pred_parts, axis=0)
    tgt_all = np.concatenate(tgt_parts, axis=0)
    counts_05 = counts_at_threshold(pred_all, tgt_all, constants.THRESHOLD_05)
    best_threshold = constants.THRESHOLD_05
    best_f1 = counts_05.f_score
    for threshold in DEFAULT_THRESHOLDS:
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


def report_as_dict(report: Slot48EvalReport, *, weights: str) -> dict[str, object]:
    """Serialize a report with ITGPT Table 2 citations.

    Args:
        report: Pooled ``M-slot48`` scores.
        weights: Checkpoint label (``last``, ``best``, ``best_slot_f1``, ``eval``).

    Returns:
        JSON-serializable mapping.
    """
    payload = report.as_dict()
    payload["citation"] = "omalley2026itgpt"
    payload["published_f1_at_05_expanded_fraxtil"] = (
        constants.PUBLISHED_F1_AT_05_EXPANDED
    )
    payload["published_f1_max_expanded_fraxtil"] = constants.PUBLISHED_F1_MAX_EXPANDED
    payload["weights"] = weights
    payload["framework"] = "pytorch"
    return payload


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
        jaccard_distance = 0.0
    else:
        inter = int(np.logical_and(base_bin, other_bin).sum())
        jaccard_distance = 1.0 - float(inter) / float(union)
    return mean_abs_delta, flip_rate, jaccard_distance


def evaluate_audio_grounding(
    model: nn.Module | SlotPredictor,
    charts: list[ddcl_datasets.DdclChart],
    *,
    threshold: float = constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
    max_beats: int = constants.MAX_BEATS,
    seed: int = DEFAULT_GROUNDING_SEED,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
    none_predictions: typing.Sequence[np.ndarray] | None = None,
) -> AudioGroundingReport:
    """Run ``none``, ``shuffle``, and ``zeros`` beat-audio grounding checks.

    Args:
        model: PyTorch ITGPT placement model or :class:`SlotPredictor`.
        charts: Loaded beat-grid charts with BPM segments and onset times.
        threshold: Occupancy cutoff for slot F1, decoded timing, and flip rates.
        thresholds: Sweep for ``slot_f1_max``.
        tolerance_sec: Match window in seconds.
        max_beats: Pad cap used at train time.
        seed: Base RNG seed for shuffling and null baselines.
        device: Optional inference device.
        amp_dtype: Optional autocast dtype (default fp32).
        none_predictions: Optional precomputed uncorrupted predictions aligned
            with ``charts`` to avoid a redundant forward pass.

    Returns:
        :class:`AudioGroundingReport` across all corruption modes.

    Raises:
        ValueError: If ``charts`` is empty, any chart exceeds ``max_beats``, or
            ``none_predictions`` length does not match ``charts``.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    if none_predictions is not None and len(none_predictions) != len(charts):
        raise ValueError(
            f"none_predictions length ({len(none_predictions)}) != "
            f"charts length ({len(charts)})"
        )
    for chart in charts:
        if chart.n_beats > max_beats:
            raise ValueError(
                f"chart {chart.song_key} has {chart.n_beats} beats > "
                f"max_beats={max_beats}"
            )
    predictor = _as_predictor(model, device=device, amp_dtype=amp_dtype)
    preds_by_kind: dict[str, list[np.ndarray]] = {}
    for kind in CORRUPTION_KINDS:
        if kind == "none" and none_predictions is not None:
            preds_by_kind[kind] = [
                np.asarray(arr, dtype=np.float32) for arr in none_predictions
            ]
        else:
            preds_by_kind[kind] = [
                predict_chart_slots(
                    predictor,
                    corrupt_chart_audio(chart, kind, seed=seed + idx),
                    max_beats=max_beats,
                )
                for idx, chart in enumerate(charts)
            ]

    tgt_all = np.concatenate([chart.slots for chart in charts], axis=0)
    rng_null = np.random.default_rng(seed)
    null_pred = shuffle_slot_null(tgt_all, rng_null)
    null_counts = counts_at_threshold(null_pred, tgt_all, threshold)

    baseline_preds = preds_by_kind["none"]
    conditions: dict[str, AudioGroundingConditionReport] = {}
    for kind in CORRUPTION_KINDS:
        kind_preds = preds_by_kind[kind]
        pred_all = np.concatenate(kind_preds, axis=0)
        slot_counts = counts_at_threshold(pred_all, tgt_all, threshold)
        best_thr = float(threshold)
        best_f1 = slot_counts.f_score
        for thr in thresholds:
            score = counts_at_threshold(pred_all, tgt_all, thr).f_score
            if score > best_f1:
                best_f1 = score
                best_thr = float(thr)

        event_tp = 0
        event_fp = 0
        event_fn = 0
        total_pred = 0
        total_ref = 0
        tm_matched = 0
        for chart, pred_slots in zip(charts, kind_preds, strict=True):
            ref_times = np.asarray(chart.gt_times_sec, dtype=np.float64)
            pred_times = (
                ddcl_slots.occupied_center_times(
                    pred_slots,
                    chart.offset_sec,
                    chart.bpm_segments,
                    threshold=threshold,
                )
                if chart.bpm_segments
                else np.empty((0,), dtype=np.float64)
            )
            counts = peak_pick.match_onsets(
                pred_times,
                ref_times,
                tolerance_sec=tolerance_sec,
            )
            event_tp += counts.true_positives
            event_fp += counts.false_positives
            event_fn += counts.false_negatives
            n_tm_matched, _ = timing_match.timing_match_counts_numpy(
                pred_times,
                ref_times,
                tolerance_sec=tolerance_sec,
            )
            tm_matched += n_tm_matched
            total_pred += int(pred_times.size)
            total_ref += int(ref_times.size)

        micro_event_f1 = peak_pick.OnsetMatchCounts(
            true_positives=event_tp,
            false_positives=event_fp,
            false_negatives=event_fn,
        ).f_score
        micro_tm = timing_match.micro_timing_match_rate(
            tm_matched,
            total_ref,
            total_pred,
        )
        mean_abs_delta, flip_rate, jaccard_dist = _prediction_shift_stats(
            baseline_preds,
            kind_preds,
            threshold=threshold,
        )
        conditions[kind] = AudioGroundingConditionReport(
            corruption=kind,
            slot_f1_at_05=slot_counts.f_score,
            slot_f1_max=best_f1,
            slot_best_threshold=best_thr,
            slot_null_f1_at_05=null_counts.f_score,
            slot_skill_f1_at_05=slot_counts.f_score - null_counts.f_score,
            slot_tp=slot_counts.true_positives,
            slot_fp=slot_counts.false_positives,
            slot_fn=slot_counts.false_negatives,
            micro_event_f1=micro_event_f1,
            micro_timing_match=micro_tm,
            n_pred=total_pred,
            n_ref=total_ref,
            mean_abs_prob_delta=mean_abs_delta,
            slot_flip_rate=flip_rate,
            occupied_jaccard_distance=jaccard_dist,
        )

    none_cond = conditions["none"]
    return AudioGroundingReport(
        threshold=float(threshold),
        tolerance_sec=float(tolerance_sec),
        n_charts=len(charts),
        n_beats=int(tgt_all.shape[0]),
        conditions=conditions,
        slot_f1_drop_shuffle=none_cond.slot_f1_at_05
        - conditions["shuffle"].slot_f1_at_05,
        slot_f1_drop_zeros=none_cond.slot_f1_at_05 - conditions["zeros"].slot_f1_at_05,
        event_f1_drop_shuffle=none_cond.micro_event_f1
        - conditions["shuffle"].micro_event_f1,
        event_f1_drop_zeros=none_cond.micro_event_f1
        - conditions["zeros"].micro_event_f1,
    )


def _slice_metrics_dict(counts: Slot48Counts) -> dict[str, object]:
    """Convert pooled counts into a compact slice metrics dictionary.

    Args:
        counts: Pooled :class:`Slot48Counts` for one slice.

    Returns:
        JSON-serializable metrics with precision, recall, F1, and counts.
    """
    n_pred = counts.true_positives + counts.false_positives
    n_ref = counts.true_positives + counts.false_negatives
    precision = float(counts.true_positives) / float(n_pred) if n_pred > 0 else 0.0
    recall = float(counts.true_positives) / float(n_ref) if n_ref > 0 else 0.0
    return {
        "f1": counts.f_score,
        "precision": precision,
        "recall": recall,
        "tp": counts.true_positives,
        "fp": counts.false_positives,
        "fn": counts.false_negatives,
        "errors": counts.false_positives + counts.false_negatives,
        "n_pred": n_pred,
        "n_ref": n_ref,
    }


def _meter_bucket(meter: int) -> str:
    """Map an integer chart meter into the canonical reporting bucket.

    Args:
        meter: Chart ``#METER`` integer.

    Returns:
        Bucket label (``<=8``, ``9-10``, ``11-12``, or ``13+``).
    """
    if meter <= 8:
        return "<=8"
    if meter <= 10:
        return "9-10"
    if meter <= 12:
        return "11-12"
    return "13+"


def _add_counts(
    bucket: dict[str, list[int]],
    key: str,
    counts: Slot48Counts,
) -> None:
    """Accumulate TP/FP/FN totals into a named slice bucket.

    Args:
        bucket: Mutable mapping of slice key to ``[tp, fp, fn]``.
        key: Slice label.
        counts: Counts to add.
    """
    acc = bucket.setdefault(key, [0, 0, 0])
    acc[0] += counts.true_positives
    acc[1] += counts.false_positives
    acc[2] += counts.false_negatives


def evaluate_slot48_slices(
    model: nn.Module | SlotPredictor,
    charts: list[ddcl_datasets.DdclChart],
    *,
    threshold: float = constants.THRESHOLD_05,
    max_beats: int = constants.MAX_BEATS,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    """Compute pooled ``M-slot48`` breakdowns by subdivision, pack, meter, and position.

    Args:
        model: PyTorch ITGPT placement model or :class:`SlotPredictor`.
        charts: Loaded beat-grid charts.
        threshold: Binary decision cutoff (default ``0.5``).
        max_beats: Pad cap used at train time.
        device: Optional inference device.
        amp_dtype: Optional autocast dtype (default fp32).

    Returns:
        JSON-serializable dictionary with ``overall``, ``by_subdivision``,
        ``by_position``, ``by_meter``, and ``by_pack`` metrics.

    Raises:
        ValueError: If ``charts`` is empty or any chart exceeds ``max_beats``.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    for chart in charts:
        if chart.n_beats > max_beats:
            raise ValueError(
                f"chart {chart.song_key} has {chart.n_beats} beats > "
                f"max_beats={max_beats}"
            )
    predictor = _as_predictor(model, device=device, amp_dtype=amp_dtype)

    subdiv_slots: dict[str, list[int]] = {
        "quarter": [0],
        "8th": [24],
        "16th_off": [12, 36],
        "24th": [8, 16, 32, 40],
    }
    covered = {idx for cols in subdiv_slots.values() for idx in cols}
    subdiv_slots["other"] = [idx for idx in range(48) if idx not in covered]

    overall = [0, 0, 0]
    by_subdiv: dict[str, list[int]] = {}
    by_pos: dict[str, list[int]] = {}
    by_meter: dict[str, list[int]] = {}
    by_pack: dict[str, list[int]] = {}

    for chart in charts:
        pred = predict_chart_slots(predictor, chart, max_beats=max_beats)
        tgt = chart.slots
        c_all = counts_at_threshold(pred, tgt, threshold)
        overall[0] += c_all.true_positives
        overall[1] += c_all.false_positives
        overall[2] += c_all.false_negatives

        for name, cols in subdiv_slots.items():
            _add_counts(
                by_subdiv,
                name,
                counts_at_threshold(pred[:, cols], tgt[:, cols], threshold),
            )

        n_beats = int(tgt.shape[0])
        intro_end = max(1, int(0.10 * n_beats))
        outro_start = max(intro_end, int(0.90 * n_beats))
        _add_counts(
            by_pos,
            "intro_0_10",
            counts_at_threshold(pred[:intro_end], tgt[:intro_end], threshold),
        )
        if outro_start > intro_end:
            _add_counts(
                by_pos,
                "body_10_90",
                counts_at_threshold(
                    pred[intro_end:outro_start],
                    tgt[intro_end:outro_start],
                    threshold,
                ),
            )
        if outro_start < n_beats:
            _add_counts(
                by_pos,
                "outro_90_100",
                counts_at_threshold(pred[outro_start:], tgt[outro_start:], threshold),
            )

        _add_counts(by_meter, _meter_bucket(int(chart.meter)), c_all)
        pack = (
            chart.song_key.split("/", 1)[0] if "/" in chart.song_key else chart.song_key
        )
        _add_counts(by_pack, pack, c_all)

    def _finalize(
        raw: dict[str, list[int]], order: tuple[str, ...] | None = None
    ) -> dict[str, dict[str, object]]:
        keys = (
            [k for k in order if k in raw] if order is not None else sorted(raw.keys())
        )
        return {
            k: _slice_metrics_dict(Slot48Counts(raw[k][0], raw[k][1], raw[k][2]))
            for k in keys
        }

    return {
        "metric": "M-slot48_slices",
        "threshold": float(threshold),
        "n_charts": len(charts),
        "n_beats": sum(int(c.n_beats) for c in charts),
        "overall": _slice_metrics_dict(
            Slot48Counts(overall[0], overall[1], overall[2])
        ),
        "by_subdivision": _finalize(
            by_subdiv, ("quarter", "8th", "16th_off", "24th", "other")
        ),
        "by_position": _finalize(by_pos, ("intro_0_10", "body_10_90", "outro_90_100")),
        "by_meter": _finalize(by_meter, ("<=8", "9-10", "11-12", "13+")),
        "by_pack": _finalize(by_pack),
    }


def _subdivision_slot_map() -> dict[str, list[int]]:
    """Return canonical 48-slot column indices grouped by rhythmic subdivision."""
    subdiv_slots: dict[str, list[int]] = {
        "quarter": [0],
        "8th": [24],
        "16th_off": [12, 36],
        "24th": [8, 16, 32, 40],
    }
    covered = {idx for cols in subdiv_slots.values() for idx in cols}
    subdiv_slots["other"] = [
        idx for idx in range(constants.N_SLOTS) if idx not in covered
    ]
    return subdiv_slots


def decode_topk_slots(
    scores: np.ndarray,
    target_count: int,
    *,
    min_score: float | None = None,
    max_score: float | None = None,
) -> np.ndarray:
    """Select the ``target_count`` highest-scoring slots as binary predictions.

    Args:
        scores: Per-slot scores ``(n_beats, 48)`` (probabilities or adjusted
            log-odds).
        target_count: Exact number of slots to activate (clamped to
            ``[0, scores.size]``).
        min_score: Optional floor: never activate slots strictly below this
            score even if ``target_count`` is not reached.
        max_score: Optional ceiling: always activate slots at or above this
            score even if ``target_count`` is exceeded.

    Returns:
        Binary float32 array of shape ``scores.shape``.
    """
    arr = np.asarray(scores, dtype=np.float64)
    out = np.zeros(arr.shape, dtype=np.float32)
    flat = arr.ravel()
    k = max(0, min(int(target_count), int(flat.size)))
    if k > 0:
        if k >= flat.size:
            mask = np.ones(flat.shape, dtype=bool)
        else:
            cutoff = float(np.partition(flat, flat.size - k)[flat.size - k])
            greater = flat > cutoff
            n_greater = int(greater.sum())
            mask = greater.copy()
            rem = k - n_greater
            if rem > 0:
                ties = np.flatnonzero(flat == cutoff)
                mask[ties[:rem]] = True
        if min_score is not None:
            mask = np.logical_and(mask, flat >= float(min_score))
        if max_score is not None:
            mask = np.logical_or(mask, flat >= float(max_score))
        out.ravel()[mask] = 1.0
    elif max_score is not None:
        out.ravel()[flat >= float(max_score)] = 1.0
    return out


def _safe_logit(probs: np.ndarray) -> np.ndarray:
    """Map probabilities in ``[0, 1]`` to numerically stable float64 logits."""
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _fit_subdivision_thresholds(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    subdiv_slots: dict[str, list[int]],
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    """Choose the F1-maximizing threshold independently for each subdivision."""
    pred_all = np.concatenate(preds, axis=0)
    tgt_all = np.concatenate(targets, axis=0)
    best_thr: dict[str, float] = {}
    for name, cols in subdiv_slots.items():
        p_col = pred_all[:, cols]
        t_col = tgt_all[:, cols]
        if int((t_col >= 0.5).sum()) == 0:
            best_thr[name] = constants.THRESHOLD_05
            continue
        best_t = constants.THRESHOLD_05
        best_f1 = counts_at_threshold(p_col, t_col, best_t).f_score
        for thr in thresholds:
            score = counts_at_threshold(p_col, t_col, thr).f_score
            if score > best_f1:
                best_f1 = score
                best_t = float(thr)
        best_thr[name] = best_t
    return best_thr


def _score_binary_predictions(
    charts: list[ddcl_datasets.DdclChart],
    bin_preds: list[np.ndarray],
    subdiv_slots: dict[str, list[int]],
    *,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
) -> dict[str, object]:
    """Score pooled slot F1, ±20 ms event F1, and subdivision/meter slices."""
    overall = [0, 0, 0]
    by_subdiv: dict[str, list[int]] = {}
    by_meter: dict[str, list[int]] = {}
    by_pack: dict[str, list[int]] = {}
    event_tp = 0
    event_fp = 0
    event_fn = 0
    total_pred = 0
    total_ref = 0

    for chart, pred_bin in zip(charts, bin_preds, strict=True):
        tgt = chart.slots
        c_all = counts_at_threshold(pred_bin, tgt, 0.5)
        overall[0] += c_all.true_positives
        overall[1] += c_all.false_positives
        overall[2] += c_all.false_negatives
        for name, cols in subdiv_slots.items():
            _add_counts(
                by_subdiv,
                name,
                counts_at_threshold(pred_bin[:, cols], tgt[:, cols], 0.5),
            )
        _add_counts(by_meter, _meter_bucket(int(chart.meter)), c_all)
        pack = (
            chart.song_key.split("/", 1)[0] if "/" in chart.song_key else chart.song_key
        )
        _add_counts(by_pack, pack, c_all)

        ref_times = np.asarray(chart.gt_times_sec, dtype=np.float64)
        pred_times = (
            ddcl_slots.occupied_center_times(
                pred_bin,
                chart.offset_sec,
                chart.bpm_segments,
                threshold=0.5,
            )
            if chart.bpm_segments
            else np.empty((0,), dtype=np.float64)
        )
        ev_counts = peak_pick.match_onsets(
            pred_times,
            ref_times,
            tolerance_sec=tolerance_sec,
        )
        event_tp += ev_counts.true_positives
        event_fp += ev_counts.false_positives
        event_fn += ev_counts.false_negatives
        total_pred += int(pred_times.size)
        total_ref += int(ref_times.size)

    def _finalize(
        raw: dict[str, list[int]], order: tuple[str, ...] | None = None
    ) -> dict[str, dict[str, object]]:
        keys = (
            [k for k in order if k in raw] if order is not None else sorted(raw.keys())
        )
        return {
            k: _slice_metrics_dict(Slot48Counts(raw[k][0], raw[k][1], raw[k][2]))
            for k in keys
        }

    overall_dict = _slice_metrics_dict(Slot48Counts(overall[0], overall[1], overall[2]))
    overall_dict["micro_event_f1"] = peak_pick.OnsetMatchCounts(
        true_positives=event_tp,
        false_positives=event_fp,
        false_negatives=event_fn,
    ).f_score
    overall_dict["density_ratio"] = (
        float(total_pred) / float(total_ref) if total_ref > 0 else 0.0
    )
    return {
        "overall": overall_dict,
        "by_subdivision": _finalize(
            by_subdiv, ("quarter", "8th", "16th_off", "24th", "other")
        ),
        "by_meter": _finalize(by_meter, ("<=8", "9-10", "11-12", "13+")),
        "by_pack": _finalize(by_pack),
    }


def evaluate_decoding_strategies(
    model: nn.Module | SlotPredictor,
    charts: list[ddcl_datasets.DdclChart],
    *,
    max_beats: int = constants.MAX_BEATS,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    tolerance_sec: float = timing_match.DEFAULT_TOLERANCE_SEC,
    device: torch.device | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, object]:
    """Compare zero-training decoding strategies with leakage-free LOSO CV.

    Evaluates five decoding methods on the exact same model probability outputs:
    ``fixed_05`` (baseline 0.5 cutoff), ``topk_density`` (exact per-chart
    target step count ``K = round(steps_per_beat * n_beats)``),
    ``topk_clamped`` (top-``K`` bounded to probabilities in ``[0.15, 0.85]``),
    ``loso_subdiv`` (leave-one-song-out per-subdivision thresholds), and
    ``loso_subdiv_topk`` (LOSO log-odds subdivision calibration + top-``K``).

    Args:
        model: Trained PyTorch placement model or :class:`SlotPredictor`.
        charts: Loaded validation charts.
        max_beats: Pad cap used at train time.
        thresholds: Candidate threshold grid for LOSO subdivision calibration.
        tolerance_sec: Time match window for ±20 ms ``micro_event_f1``.
        device: Optional inference device.
        amp_dtype: Optional autocast dtype (default fp32).

    Returns:
        JSON-serializable comparison dictionary keyed by decoding strategy.

    Raises:
        ValueError: If ``charts`` is empty or any chart exceeds ``max_beats``.
    """
    if not charts:
        raise ValueError("charts must be non-empty")
    for chart in charts:
        if chart.n_beats > max_beats:
            raise ValueError(
                f"chart {chart.song_key} has {chart.n_beats} beats > "
                f"max_beats={max_beats}"
            )
    predictor = _as_predictor(model, device=device, amp_dtype=amp_dtype)
    subdiv_slots = _subdivision_slot_map()
    preds = [
        predict_chart_slots(predictor, chart, max_beats=max_beats) for chart in charts
    ]
    targets = [chart.slots for chart in charts]
    target_counts = [int(round(float(chart.slots.sum()))) for chart in charts]

    fixed_bins = [(p >= constants.THRESHOLD_05).astype(np.float32) for p in preds]
    topk_bins = [
        decode_topk_slots(p, k) for p, k in zip(preds, target_counts, strict=True)
    ]
    topk_clamped_bins = [
        decode_topk_slots(p, k, min_score=0.15, max_score=0.85)
        for p, k in zip(preds, target_counts, strict=True)
    ]

    song_keys = [chart.song_key for chart in charts]
    unique_songs = sorted(set(song_keys))
    loso_subdiv_bins: list[np.ndarray] = [
        np.zeros_like(p, dtype=np.float32) for p in preds
    ]
    loso_subdiv_topk_bins: list[np.ndarray] = [
        np.zeros_like(p, dtype=np.float32) for p in preds
    ]
    fold_thresholds: dict[str, list[float]] = {k: [] for k in subdiv_slots}

    for held_song in unique_songs:
        train_idx = [i for i, key in enumerate(song_keys) if key != held_song]
        eval_idx = [i for i, key in enumerate(song_keys) if key == held_song]
        if train_idx:
            sub_thr = _fit_subdivision_thresholds(
                [preds[i] for i in train_idx],
                [targets[i] for i in train_idx],
                subdiv_slots,
                thresholds,
            )
        else:
            sub_thr = dict.fromkeys(subdiv_slots, constants.THRESHOLD_05)
        for name, val in sub_thr.items():
            fold_thresholds[name].append(float(val))
        thr_vec = np.full(
            (constants.N_SLOTS,), constants.THRESHOLD_05, dtype=np.float64
        )
        for name, cols in subdiv_slots.items():
            thr_vec[cols] = float(sub_thr[name])
        thr_logits = _safe_logit(thr_vec)
        for i in eval_idx:
            loso_subdiv_bins[i] = (preds[i] >= thr_vec[None, :]).astype(np.float32)
            adjusted_scores = _safe_logit(preds[i]) - thr_logits[None, :]
            loso_subdiv_topk_bins[i] = decode_topk_slots(
                adjusted_scores, target_counts[i]
            )

    mean_subdiv_thr = {
        name: float(np.mean(vals)) if vals else constants.THRESHOLD_05
        for name, vals in fold_thresholds.items()
    }
    return {
        "metric": "M-slot48_decoding_strategies",
        "n_charts": len(charts),
        "n_songs": len(unique_songs),
        "n_beats": sum(int(c.n_beats) for c in charts),
        "loso_mean_subdiv_thresholds": mean_subdiv_thr,
        "strategies": {
            "fixed_05": _score_binary_predictions(
                charts, fixed_bins, subdiv_slots, tolerance_sec=tolerance_sec
            ),
            "topk_density": _score_binary_predictions(
                charts, topk_bins, subdiv_slots, tolerance_sec=tolerance_sec
            ),
            "topk_clamped": _score_binary_predictions(
                charts, topk_clamped_bins, subdiv_slots, tolerance_sec=tolerance_sec
            ),
            "loso_subdiv": _score_binary_predictions(
                charts, loso_subdiv_bins, subdiv_slots, tolerance_sec=tolerance_sec
            ),
            "loso_subdiv_topk": _score_binary_predictions(
                charts, loso_subdiv_topk_bins, subdiv_slots, tolerance_sec=tolerance_sec
            ),
        },
    }
