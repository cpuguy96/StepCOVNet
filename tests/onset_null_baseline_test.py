"""Tests for audio-blind null baselines on onset timing metrics."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet import onset_null_baseline


def _dense_gt(n: int = 600, scale: float = 1.0) -> np.ndarray:
    """Onset times with realistic spacing: dense, grid-quantized, non-periodic.

    Mean inter-onset interval is ~180 ms at ``scale=1``, matching the measured
    ladder val set. A pure metronome must not reproduce it, or the baselines
    under test would be scored against a degenerate fixture.
    """
    rng = np.random.default_rng(20260804)
    choices = np.arange(0.08, 0.42, 0.02)
    weights = np.linspace(3.0, 1.0, choices.size)
    weights = weights / weights.sum()
    iois = rng.choice(choices, size=n - 1, p=weights) * scale
    return np.concatenate([[1.0], 1.0 + np.cumsum(iois)])


class BuildNullOnsetsTest(unittest.TestCase):
    def test_emits_requested_count_sorted_and_in_range(self) -> None:
        gt = _dense_gt()
        duration = float(gt[-1]) + 5.0
        for kind in onset_null_baseline.DEFAULT_KINDS:
            with self.subTest(kind=kind):
                pred = onset_null_baseline.build_null_onsets(
                    kind,
                    gt,
                    duration_sec=duration,
                    n_pred=250,
                    rng=np.random.default_rng(0),
                )
                self.assertEqual(pred.size, 250)
                self.assertTrue(np.all(np.diff(pred) >= 0.0))
                self.assertGreaterEqual(float(pred.min()), 0.0)
                self.assertLessEqual(float(pred.max()), duration)

    def test_snaps_to_prediction_grid(self) -> None:
        gt = _dense_gt()
        pred = onset_null_baseline.build_null_onsets(
            "uniform_duration",
            gt,
            duration_sec=120.0,
            n_pred=64,
            rng=np.random.default_rng(1),
            hop_sec=0.02,
        )
        residual = np.abs(pred / 0.02 - np.round(pred / 0.02))
        self.assertTrue(np.all(residual < 1e-6))

    def test_is_reproducible_for_a_seed(self) -> None:
        gt = _dense_gt()
        first = onset_null_baseline.build_null_onsets(
            "ioi_shuffle",
            gt,
            duration_sec=200.0,
            n_pred=100,
            rng=np.random.default_rng(7),
        )
        second = onset_null_baseline.build_null_onsets(
            "ioi_shuffle",
            gt,
            duration_sec=200.0,
            n_pred=100,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_allclose(first, second)

    def test_empty_when_no_predictions_requested(self) -> None:
        pred = onset_null_baseline.build_null_onsets(
            "regular_grid",
            _dense_gt(),
            duration_sec=120.0,
            n_pred=0,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(pred.size, 0)

    def test_rejects_unknown_kind(self) -> None:
        with self.assertRaises(ValueError):
            onset_null_baseline.build_null_onsets(
                "not_a_baseline",
                _dense_gt(),
                duration_sec=120.0,
                n_pred=10,
                rng=np.random.default_rng(0),
            )


class NullFloorTest(unittest.TestCase):
    def test_dense_charts_give_hungarian_f1_a_high_chance_floor(self) -> None:
        gt = _dense_gt(n=600)
        counts = onset_null_baseline.null_counts_for_song(
            gt,
            duration_sec=float(gt[-1]) + 2.0,
            n_pred=gt.size,
            tolerance_sec=0.02,
        )
        aggregated = onset_null_baseline.aggregate_null_counts([counts])
        _, floor = onset_null_baseline.strongest_null(aggregated)
        self.assertGreater(floor, 0.15)

    def test_ordered_timing_match_floor_stays_near_zero(self) -> None:
        gt = _dense_gt(n=600)
        counts = onset_null_baseline.null_counts_for_song(
            gt,
            duration_sec=float(gt[-1]) + 2.0,
            n_pred=gt.size,
            tolerance_sec=0.02,
        )
        aggregated = onset_null_baseline.aggregate_null_counts([counts])
        _, floor = onset_null_baseline.strongest_null(
            aggregated,
            metric="timing_match",
        )
        self.assertLess(floor, 0.05)

    def test_floor_falls_when_fewer_predictions_are_emitted(self) -> None:
        gt = _dense_gt()
        duration = float(gt[-1]) + 2.0
        floors = []
        for ratio in (0.2, 1.0):
            counts = onset_null_baseline.null_counts_for_song(
                gt,
                duration_sec=duration,
                n_pred=int(ratio * gt.size),
                tolerance_sec=0.02,
            )
            aggregated = onset_null_baseline.aggregate_null_counts([counts])
            floors.append(onset_null_baseline.strongest_null(aggregated)[1])
        self.assertLess(floors[0], floors[1])

    def test_sparse_charts_have_a_low_floor(self) -> None:
        gt = _dense_gt(n=40, scale=30.0)
        counts = onset_null_baseline.null_counts_for_song(
            gt,
            duration_sec=float(gt[-1]) + 2.0,
            n_pred=gt.size,
            tolerance_sec=0.02,
        )
        aggregated = onset_null_baseline.aggregate_null_counts([counts])
        _, floor = onset_null_baseline.strongest_null(aggregated)
        self.assertLess(floor, 0.15)


class SkillOverNullTest(unittest.TestCase):
    def test_zero_when_score_equals_the_floor(self) -> None:
        self.assertAlmostEqual(onset_null_baseline.skill_over_null(0.24, 0.24), 0.0)

    def test_one_for_a_perfect_score(self) -> None:
        self.assertAlmostEqual(onset_null_baseline.skill_over_null(1.0, 0.24), 1.0)

    def test_negative_when_the_baseline_wins(self) -> None:
        self.assertLess(onset_null_baseline.skill_over_null(0.20, 0.24), 0.0)

    def test_zero_when_the_floor_leaves_no_headroom(self) -> None:
        self.assertAlmostEqual(onset_null_baseline.skill_over_null(0.5, 1.0), 0.0)


class AggregateTest(unittest.TestCase):
    def test_micro_averages_across_songs(self) -> None:
        gt = _dense_gt(n=300)
        duration = float(gt[-1]) + 2.0
        rows = [
            onset_null_baseline.null_counts_for_song(
                gt,
                duration_sec=duration,
                n_pred=gt.size,
                tolerance_sec=0.02,
            )
            for _ in range(3)
        ]
        aggregated = onset_null_baseline.aggregate_null_counts(rows)
        self.assertEqual(
            set(aggregated),
            set(onset_null_baseline.DEFAULT_KINDS),
        )
        for vals in aggregated.values():
            total = (
                int(vals["true_positives"])
                + int(vals["false_positives"])
                + int(vals["false_negatives"])
            )
            self.assertGreater(total, 0)

    def test_strongest_null_picks_the_highest_floor(self) -> None:
        aggregated = {
            "a": {"event_f1": 0.1, "timing_match": 0.9},
            "b": {"event_f1": 0.3, "timing_match": 0.0},
        }
        self.assertEqual(
            onset_null_baseline.strongest_null(aggregated),
            ("b", 0.3),
        )
        self.assertEqual(
            onset_null_baseline.strongest_null(aggregated, metric="timing_match"),
            ("a", 0.9),
        )


if __name__ == "__main__":
    unittest.main()
