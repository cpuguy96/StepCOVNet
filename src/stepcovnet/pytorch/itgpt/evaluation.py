"""``M-slot48`` evaluation and audio-grounding for PyTorch ITGPT (zero TF/Keras)."""

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
            verbose: Unused compatibility parameter matching Keras ``Model.predict``.

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
    cap = (
        max(datasets.pad_length(chart.n_beats, chart.n_beats), constants.CHUNK_ALIGN)
        if max_beats is None
        else max(int(max_beats), constants.CHUNK_ALIGN)
    )
    inputs, _, mask = datasets.pack_chart(chart, max_beats=cap)
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
    rng = np.random.default_rng(seed)
    pred_parts: list[np.ndarray] = []
    tgt_parts: list[np.ndarray] = []
    for chart in charts:
        inputs, _, mask = datasets.pack_chart(chart, max_beats=max_beats)
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

    Returns:
        :class:`AudioGroundingReport` across all corruption modes.

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
    preds_by_kind: dict[str, list[np.ndarray]] = {}
    for kind in CORRUPTION_KINDS:
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
