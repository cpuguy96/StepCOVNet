"""Shared ``M-slot48`` binary F1 and audio-corruption helpers (`omalley2025ddcl`, `omalley2026itgpt`).

Pure NumPy scoring primitives shared across DDCL, ITGPT, and DDC peak-snapped
``M-slot48`` evaluation. Zero PyTorch/TensorFlow model coupling.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Sequence

import numpy as np

from stepcovnet.ddc import peak_pick
from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets

DEFAULT_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))
CORRUPTION_KINDS = ("none", "shuffle", "zeros")
DEFAULT_GROUNDING_SEED = 0


class SlotPredictor(typing.Protocol):
    """Minimal predict interface used by ``M-slot48`` evaluation."""

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


def sweep_slot_counts(
    pred_all: np.ndarray,
    tgt_all: np.ndarray,
    *,
    threshold: float = ddcl_constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> tuple[Slot48Counts, float, float]:
    """Compute counts at ``threshold`` and max-F1 across ``thresholds``.

    Args:
        pred_all: Concatenated predicted probabilities ``(total_beats, 48)``.
        tgt_all: Concatenated ground-truth 0/1 slots ``(total_beats, 48)``.
        threshold: Primary cutoff for ``counts_primary``.
        thresholds: Candidate cutoffs swept for ``best_f1``.

    Returns:
        Tuple ``(counts_primary, best_f1, best_threshold)``.
    """
    primary = float(threshold)
    counts_primary = counts_at_threshold(pred_all, tgt_all, primary)
    best_threshold = primary
    best_f1 = counts_primary.f_score
    for tau in thresholds:
        tau_f = float(tau)
        score = counts_at_threshold(pred_all, tgt_all, tau_f).f_score
        if score > best_f1:
            best_f1 = score
            best_threshold = tau_f
    return counts_primary, best_f1, best_threshold


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
    """Pooled ``M-slot48`` scores across DDCL and ITGPT placement models.

    Attributes:
        f1_at_05: Micro F1 at threshold 0.5.
        f1_max: Best micro F1 over the threshold sweep.
        best_threshold: Threshold that achieved ``f1_max``.
        null_f1_at_05: Beat-shuffled GT scored at 0.5 against real GT.
        n_charts: Charts evaluated.
        n_beats: Pooled beat count.
        counts_at_05: Pooled counts at 0.5.
        citation: Paper citation key emitted by :meth:`as_dict`.
        published_f1_at_05_expanded_fraxtil: Published baseline F1@0.5.
        published_f1_max_expanded_fraxtil: Published baseline max-F1.
    """

    f1_at_05: float
    f1_max: float
    best_threshold: float
    null_f1_at_05: float
    n_charts: int
    n_beats: int
    counts_at_05: Slot48Counts
    citation: str = "omalley2025ddcl"
    published_f1_at_05_expanded_fraxtil: float = (
        ddcl_constants.PUBLISHED_F1_AT_05_EXPANDED
    )
    published_f1_max_expanded_fraxtil: float = ddcl_constants.PUBLISHED_F1_MAX_EXPANDED

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
                self.published_f1_at_05_expanded_fraxtil
            ),
            "published_f1_max_expanded_fraxtil": (
                self.published_f1_max_expanded_fraxtil
            ),
            "citation": self.citation,
        }


def score_slot48_predictions(
    pred_parts: Sequence[np.ndarray],
    tgt_parts: Sequence[np.ndarray],
    *,
    seed: int = 42,
    threshold_primary: float = ddcl_constants.THRESHOLD_05,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    citation: str = "omalley2025ddcl",
    published_f1_at_05: float = ddcl_constants.PUBLISHED_F1_AT_05_EXPANDED,
    published_f1_max: float = ddcl_constants.PUBLISHED_F1_MAX_EXPANDED,
) -> Slot48EvalReport:
    """Pool precomputed per-chart slot predictions into a :class:`Slot48EvalReport`.

    Args:
        pred_parts: Per-chart predicted probabilities ``(n_beats, 48)``.
        tgt_parts: Per-chart ground-truth 0/1 slot matrices ``(n_beats, 48)``.
        seed: RNG seed for beat-shuffle null baseline.
        threshold_primary: Primary threshold (default 0.5).
        thresholds: Candidate threshold sweep for ``f1_max``.
        citation: Paper citation key to store on the report.
        published_f1_at_05: Published reference F1@0.5.
        published_f1_max: Published reference max-F1.

    Returns:
        Pooled :class:`Slot48EvalReport`.

    Raises:
        ValueError: If ``pred_parts`` is empty or differs in length from ``tgt_parts``.
    """
    if not pred_parts:
        raise ValueError("pred_parts must be non-empty")
    if len(pred_parts) != len(tgt_parts):
        raise ValueError(
            f"pred_parts length ({len(pred_parts)}) != "
            f"tgt_parts length ({len(tgt_parts)})"
        )
    rng = np.random.default_rng(seed)
    pred_all = np.concatenate(pred_parts, axis=0)
    tgt_all = np.concatenate(tgt_parts, axis=0)
    counts_05, best_f1, best_threshold = sweep_slot_counts(
        pred_all,
        tgt_all,
        threshold=threshold_primary,
        thresholds=thresholds,
    )
    null_pred = shuffle_slot_null(tgt_all, rng)
    null_counts = counts_at_threshold(null_pred, tgt_all, threshold_primary)
    return Slot48EvalReport(
        f1_at_05=counts_05.f_score,
        f1_max=best_f1,
        best_threshold=best_threshold,
        null_f1_at_05=null_counts.f_score,
        n_charts=len(pred_parts),
        n_beats=int(tgt_all.shape[0]),
        counts_at_05=counts_05,
        citation=citation,
        published_f1_at_05_expanded_fraxtil=published_f1_at_05,
        published_f1_max_expanded_fraxtil=published_f1_max,
    )


def report_as_dict(
    report: Slot48EvalReport,
    *,
    weights: str,
    citation: str,
    published_f1_at_05: float,
    published_f1_max: float,
) -> dict[str, object]:
    """Serialize a :class:`Slot48EvalReport` with citation, baseline, and tag metadata.

    Args:
        report: Pooled ``M-slot48`` scores.
        weights: Checkpoint label (``last``, ``best``, ``best_slot_f1``, ``eval``).
        citation: Paper citation key.
        published_f1_at_05: Published reference F1@0.5.
        published_f1_max: Published reference max-F1.

    Returns:
        JSON-serializable dictionary matching DDCL/ITGPT eval JSON outputs.
    """
    payload = report.as_dict()
    payload["citation"] = citation
    payload["published_f1_at_05_expanded_fraxtil"] = published_f1_at_05
    payload["published_f1_max_expanded_fraxtil"] = published_f1_max
    payload["weights"] = weights
    payload["framework"] = "pytorch"
    return payload


def corrupt_chart_audio(
    chart: ddcl_datasets.DdclChart,
    corruption: str,
    *,
    seed: int = DEFAULT_GROUNDING_SEED,
) -> ddcl_datasets.DdclChart:
    """Return a chart copy with ``beat_audio`` corrupted and metadata intact.

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


def prediction_shift_stats(
    baseline_preds: Sequence[np.ndarray],
    other_preds: Sequence[np.ndarray],
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
        intersection = int(np.logical_and(base_bin, other_bin).sum())
        jaccard_distance = 1.0 - float(intersection) / float(union)
    return mean_abs_delta, flip_rate, jaccard_distance
