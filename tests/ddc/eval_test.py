"""Tests for DDC placement eval (`M-ddc-20ms`, PyTorch)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from stepcovnet.ddc import datasets, evaluation


class _QueueModel(nn.Module):
    """Stub PyTorch module that returns queued salience traces."""

    def __init__(self, traces: list[np.ndarray]) -> None:
        super().__init__()
        self._traces = list(traces)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the next queued trace as ``(1, time, 1)``."""
        del inputs
        trace = self._traces.pop(0)
        return torch.as_tensor(trace, dtype=torch.float32).reshape(1, -1, 1)


def _chart_with_onsets(*, difficulty: str = "hard") -> datasets.PlacementChart:
    """Return a short chart with two onsets."""
    n_frames = 40
    target = np.zeros((n_frames,), dtype=np.float32)
    target[10] = 1.0
    target[20] = 1.0
    spec = np.zeros((n_frames, 80, 3), dtype=np.float32)
    return datasets.PlacementChart(
        song_key="bundle/song",
        difficulty=difficulty,
        spec=spec,
        target=target,
        gt_times=np.array([0.10, 0.20], dtype=np.float32),
        first_onset=10,
        last_onset=20,
    )


class DdcEvalTest(unittest.TestCase):
    def test_perfect_salience_scores_high(self) -> None:
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
        self.assertEqual(payload["framework"], "pytorch")

    def test_evaluate_empty_charts_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluation.evaluate_placement(_QueueModel([]), [])

    def test_threshold_selection_across_difficulties(self) -> None:
        hard = _chart_with_onsets(difficulty="hard")
        easy = _chart_with_onsets(difficulty="easy")
        model = _QueueModel([hard.target.copy(), easy.target.copy()])
        report = evaluation.evaluate_placement(
            model,
            [hard, easy],
            thresholds=(0.1, 0.5, 0.9),
            seed=0,
        )
        self.assertEqual(set(report.per_difficulty_threshold), {"hard", "easy"})
        self.assertIn(report.per_difficulty_threshold["hard"], (0.1, 0.5, 0.9))

        fixed_model = _QueueModel([hard.target.copy()])
        fixed = evaluation.evaluate_placement(
            fixed_model,
            [hard],
            thresholds=(0.5,),
            seed=0,
        )
        self.assertEqual(fixed.per_difficulty_threshold["hard"], 0.5)

    def test_extra_peak_lowers_timing_match_below_greedy_f1(self) -> None:
        chart = _chart_with_onsets()
        salience = chart.target.copy()
        salience[30] = 0.6
        model = _QueueModel([salience])
        report = evaluation.evaluate_placement(
            model,
            [chart],
            thresholds=(0.5,),
            seed=0,
        )
        self.assertEqual(report.counts.true_positives, 2)
        self.assertEqual(report.counts.false_positives, 1)
        self.assertGreater(report.f_score_m, 0.7)
        self.assertAlmostEqual(report.timing_match, 2.0 / 3.0)
        self.assertLess(report.timing_match, report.f_score_m)


if __name__ == "__main__":
    unittest.main()
