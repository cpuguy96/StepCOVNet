"""``M-slot48`` binary F1 for DDCL placement (`omalley2025ddcl`, `omalley2026itgpt`).

Reports micro F1 at threshold 0.5 and the max-F1 threshold sweep used in the
ITGPT comparison table. This is **not** Hamming ±20 ms (`M-ddc-20ms`).
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

from stepcovnet.ddc import peak_pick
from stepcovnet.ddcl import constants, datasets

DEFAULT_THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))


class SlotPredictor(Protocol):
    """Minimal predict interface used by ``M-slot48`` eval."""

    def predict(self, inputs: object, verbose: int = 0) -> object:
        """Return slot probabilities.

        Args:
            inputs: Keras input dict.
            verbose: Keras verbosity.

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
        inputs = datasets.chart_model_inputs(chart)
        pred = np.asarray(model.predict(inputs, verbose=0), dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        pred_parts.append(pred)
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


def predict_chart_slots(model: SlotPredictor, chart: datasets.DdclChart) -> np.ndarray:
    """Return slot probabilities for one chart.

    Args:
        model: Trained placement model.
        chart: Loaded chart.

    Returns:
        Probabilities ``(n_beats, 48)``.
    """
    inputs = datasets.chart_model_inputs(chart)
    pred = np.asarray(model.predict(inputs, verbose=0), dtype=np.float32)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    return pred
