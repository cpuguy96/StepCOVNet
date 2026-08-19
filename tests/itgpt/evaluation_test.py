"""Tests for ITGPT ``M-slot48`` evaluation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.ddcl import datasets as ddcl_datasets
from stepcovnet.itgpt import constants, evaluation


class _ConstPredictor:
    """Return a constant slot probability tensor."""

    def __init__(self, value: float, *, squeeze_batch: bool = False) -> None:
        """Store the fill value.

        Args:
            value: Probability written to every slot.
            squeeze_batch: If True, return ``(beats, 48)``.
        """
        self.value = value
        self.squeeze_batch = squeeze_batch

    def predict(self, inputs: object, verbose: int = 0) -> np.ndarray:
        """Return filled slot probabilities.

        Args:
            inputs: Packed chart dict with ``audio``.
            verbose: Unused Keras verbosity.

        Returns:
            Constant probabilities.
        """
        del verbose
        n_beats = int(inputs["audio"].shape[1])  # type: ignore[index]
        pred = np.full((1, n_beats, constants.N_SLOTS), self.value, dtype=np.float32)
        if self.squeeze_batch:
            return pred[0]
        return pred


def _chart(*, n_beats: int = 8) -> ddcl_datasets.DdclChart:
    """Build a tiny labeled chart.

    Args:
        n_beats: Integer-beat length.

    Returns:
        Chart with one occupied 16th slot.
    """
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[0, 0] = 1.0
    stream = np.zeros((n_beats, ddcl_constants.STREAM_DIM), dtype=np.float32)
    stream[:, 1] = 140.0
    return ddcl_datasets.DdclChart(
        song_key="bundle/song",
        difficulty="easy",
        meter=5,
        beat_audio=np.zeros(
            (
                n_beats,
                constants.N_FRAMES_PER_BEAT,
                constants.N_MELS,
                constants.N_CHANNELS,
            ),
            dtype=np.float32,
        ),
        stream=stream,
        slots=slots,
        memlen=0,
    )


class ItgptEvaluationTest(unittest.TestCase):
    def test_evaluate_slot48_rejects_empty(self):
        with self.assertRaises(ValueError):
            evaluation.evaluate_slot48(_ConstPredictor(0.9), [])

    def test_predict_and_report_cite_table2(self):
        chart = _chart()
        pred = evaluation.predict_chart_slots(
            _ConstPredictor(0.9, squeeze_batch=True), chart
        )
        self.assertEqual(pred.shape, (chart.n_beats, constants.N_SLOTS))
        report = evaluation.evaluate_slot48(
            _ConstPredictor(0.9),
            [chart],
            max_beats=constants.CHUNK_ALIGN,
        )
        self.assertGreater(report.f1_at_05, 0.0)
        mixed = evaluation.evaluate_slot48(
            _ConstPredictor(0.2),
            [chart],
            max_beats=constants.CHUNK_ALIGN,
        )
        self.assertGreater(mixed.f1_max, mixed.f1_at_05)
        payload = evaluation.report_as_dict(report, weights="last")
        self.assertEqual(payload["citation"], "omalley2026itgpt")
        self.assertEqual(payload["weights"], "last")
        self.assertEqual(
            payload["published_f1_at_05_expanded_fraxtil"],
            constants.PUBLISHED_F1_AT_05_EXPANDED,
        )


if __name__ == "__main__":
    unittest.main()
