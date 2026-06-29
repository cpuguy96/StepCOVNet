"""Tests for unified onset timing_match metric."""

import unittest

import numpy as np

from stepcovnet import timing_match


class TimingMatchTest(unittest.TestCase):
    def test_timing_match_counts_exact_and_short(self) -> None:
        tol = 0.02
        pred = np.array([0.0, 0.05, 0.10], dtype=np.float64)
        ref = np.array([0.0, 0.04, 0.10], dtype=np.float64)
        self.assertEqual(
            timing_match.timing_match_counts_numpy(pred, ref, tolerance_sec=tol),
            (3, 3),
        )
        self.assertEqual(
            timing_match.timing_match_counts_numpy(
                pred[:2],
                ref,
                tolerance_sec=tol,
            ),
            (2, 3),
        )

    def test_timing_match_rate_penalizes_extra_predictions(self) -> None:
        tol = 0.02
        pred = np.array([0.0, 0.04, 0.10, 0.20], dtype=np.float64)
        ref = np.array([0.0, 0.04, 0.10], dtype=np.float64)
        self.assertAlmostEqual(
            timing_match.timing_match_rate_numpy(pred, ref, tolerance_sec=tol),
            3.0 / 4.0,
        )

    def test_timing_match_rate_missing_predictions(self) -> None:
        tol = 0.02
        pred = np.array([0.0, 0.04], dtype=np.float64)
        ref = np.array([0.0, 0.04, 0.10], dtype=np.float64)
        self.assertAlmostEqual(
            timing_match.timing_match_rate_numpy(pred, ref, tolerance_sec=tol),
            2.0 / 3.0,
        )

    def test_timing_match_rate_empty_ref(self) -> None:
        self.assertEqual(
            timing_match.timing_match_rate_numpy(
                np.array([1.0]),
                np.array([]),
                tolerance_sec=0.02,
            ),
            0.0,
        )

    def test_reference_times_from_mask_sorts_and_filters(self) -> None:
        times = np.array([0.5, 0.1, 0.3, 0.0], dtype=np.float32)
        mask = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        kept = timing_match.reference_times_from_mask(times, mask)
        np.testing.assert_allclose(kept, [0.0, 0.3, 0.5])

    def test_timing_match_report_fields(self) -> None:
        report = timing_match.timing_match_report(
            np.array([0.0, 0.02]),
            np.array([0.0, 0.05]),
            tolerance_sec=0.02,
        )
        self.assertEqual(report["n_matched"], 1)
        self.assertEqual(report["n_pred"], 2)
        self.assertEqual(report["n_ref"], 2)
        self.assertEqual(report["n_denom"], 2)
        self.assertAlmostEqual(report["rate"], 0.5)
        self.assertAlmostEqual(report["tolerance_sec"], 0.02)

    def test_micro_timing_match_rate(self) -> None:
        self.assertAlmostEqual(
            timing_match.micro_timing_match_rate(633.0, 634.0, 634.0),
            633.0 / 634.0,
        )
        self.assertAlmostEqual(
            timing_match.micro_timing_match_rate(634.0, 634.0, 635.0),
            634.0 / 635.0,
        )
        self.assertEqual(
            timing_match.micro_timing_match_rate(0.0, 0.0, 0.0),
            0.0,
        )

    def test_ordered_aliases_match_canonical(self) -> None:
        pred = np.array([1.0, 2.0])
        ref = np.array([1.01, 2.0])
        self.assertEqual(
            timing_match.ordered_onset_match_counts_numpy(
                pred,
                ref,
                tolerance_sec=0.02,
            ),
            timing_match.timing_match_counts_numpy(
                pred,
                ref,
                tolerance_sec=0.02,
            ),
        )


if __name__ == "__main__":
    unittest.main()
