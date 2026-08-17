"""Tests for DDCL ``M-slot48`` evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddcl import datasets, evaluation


class _StubModel:
    """Keras-like model that returns a queued slot matrix."""

    def __init__(self, pred: np.ndarray) -> None:
        self._pred = np.asarray(pred, dtype=np.float32)

    def predict(self, inputs, verbose=0):
        """Return the queued predictions.

        Args:
            inputs: Unused.
            verbose: Unused Keras flag.

        Returns:
            Slot probabilities.
        """
        del inputs, verbose
        return self._pred


def _tiny_chart(slots: np.ndarray) -> datasets.DdclChart:
    """Return a chart with dummy audio windows matching ``slots``.

    Args:
        slots: Target ``(n_beats, 48)``.

    Returns:
        DdclChart.
    """
    n_beats = slots.shape[0]
    beat_audio = np.zeros((n_beats, 4, 4, 3), dtype=np.float32)
    return datasets.DdclChart(
        song_key="pack/song",
        difficulty="challenge",
        meter=9,
        beat_audio=beat_audio,
        stream=np.zeros((n_beats, 2), dtype=np.float32),
        slots=slots.astype(np.float32),
        memlen=1,
    )


class DdclEvalTest(unittest.TestCase):
    def test_perfect_prediction_beats_null(self):
        target = np.zeros((8, 48), dtype=np.float32)
        target[0, 0] = 1.0
        target[3, 24] = 1.0
        target[7, 47] = 1.0
        chart = _tiny_chart(target)
        model = _StubModel(target)
        report = evaluation.evaluate_slot48(model, [chart], seed=0)
        self.assertAlmostEqual(report.f1_at_05, 1.0)
        self.assertAlmostEqual(report.f1_max, 1.0)
        self.assertLess(report.null_f1_at_05, 1.0)
        self.assertEqual(report.as_dict()["metric"], "M-slot48")

    def test_counts_at_threshold(self):
        pred = np.array([[0.9, 0.1, 0.0]])
        target = np.array([[1.0, 0.0, 1.0]])
        counts = evaluation.counts_at_threshold(pred, target, 0.5)
        self.assertEqual(counts.true_positives, 1)
        self.assertEqual(counts.false_positives, 0)
        self.assertEqual(counts.false_negatives, 1)


if __name__ == "__main__":
    unittest.main()
