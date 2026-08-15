"""Tests for DDC placement eval (`M-ddc-20ms`)."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddc import datasets, evaluation


class _QueueModel:
    """Stub Keras-like model that returns queued salience traces."""

    def __init__(self, traces: list[np.ndarray]) -> None:
        self._traces = list(traces)

    def predict(self, inputs, verbose=0):
        """Return the next queued trace.

        Args:
            inputs: Unused model inputs.
            verbose: Unused Keras flag.

        Returns:
            Salience with shape ``(1, time, 1)``.
        """
        del inputs, verbose
        trace = self._traces.pop(0)
        return np.asarray(trace, dtype=np.float32).reshape(1, -1, 1)


def _chart_with_onsets() -> datasets.PlacementChart:
    """Return a short chart with two onsets.

    Returns:
        Placement chart.
    """
    n_frames = 40
    target = np.zeros((n_frames,), dtype=np.float32)
    target[10] = 1.0
    target[20] = 1.0
    spec = np.zeros((n_frames, 80, 3), dtype=np.float32)
    return datasets.PlacementChart(
        song_key="bundle/song",
        difficulty="hard",
        spec=spec,
        target=target,
        gt_times=np.array([0.10, 0.20], dtype=np.float32),
        first_onset=10,
        last_onset=20,
    )


class DdcEvalTest(unittest.TestCase):
    def test_perfect_salience_scores_high(self):
        chart = _chart_with_onsets()
        salience = chart.target.copy()
        model = _QueueModel([salience])
        report = evaluation.evaluate_placement(model, [chart], seed=0)
        self.assertEqual(report.n_charts, 1)
        self.assertGreater(report.f_score_m, 0.9)
        self.assertIn("hard", report.per_difficulty_threshold)
        payload = report.as_dict()
        self.assertIn("skill_f_score_m", payload)
        self.assertGreater(report.timing_match, 0.9)
        self.assertIn("timing_match", payload)
        self.assertIn("null_timing_match", payload)
        self.assertIn("skill_timing_match", payload)
        self.assertEqual(payload["timing_match_tolerance_sec"], 0.02)

    def test_evaluate_empty_charts_raises(self):
        with self.assertRaises(ValueError):
            evaluation.evaluate_placement(_QueueModel([]), [])

    def test_choose_thresholds_prefers_working_value(self):
        chart = _chart_with_onsets()
        salience = chart.target.copy()
        chosen = evaluation.choose_thresholds([salience], [chart], grid=(0.1, 0.9))
        self.assertIn(chosen["hard"], (0.1, 0.9))
        model = _QueueModel([chart.target.copy()])
        report = evaluation.evaluate_placement(
            model,
            [chart],
            thresholds={"hard": 0.5},
            tune_on=False,
            seed=0,
        )
        self.assertEqual(report.per_difficulty_threshold["hard"], 0.5)
        easy = _chart_with_onsets()
        easy.difficulty = "easy"
        chosen_multi = evaluation.choose_thresholds(
            [salience, salience],
            [chart, easy],
            grid=(0.5,),
        )
        self.assertEqual(set(chosen_multi), {"hard", "easy"})
        model2 = _QueueModel([chart.target.copy()])
        untuned = evaluation.evaluate_placement(
            model2,
            [chart],
            tune_on=False,
            seed=0,
        )
        self.assertEqual(untuned.per_difficulty_threshold["hard"], 0.5)

    def test_extra_peak_lowers_timing_match_not_greedy_f1(self):
        chart = _chart_with_onsets()
        salience = chart.target.copy()
        salience[30] = 1.0
        model = _QueueModel([salience])
        report = evaluation.evaluate_placement(
            model,
            [chart],
            thresholds={"hard": 0.5},
            tune_on=False,
            seed=0,
        )
        self.assertEqual(report.counts.true_positives, 2)
        self.assertEqual(report.counts.false_positives, 1)
        self.assertGreater(report.f_score_m, 0.7)
        self.assertAlmostEqual(report.timing_match, 2.0 / 3.0)
        self.assertLess(report.timing_match, report.f_score_m)


if __name__ == "__main__":
    unittest.main()
