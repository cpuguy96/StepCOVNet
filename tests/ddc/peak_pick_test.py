"""Tests for DDC Hamming peak-pick (`M-ddc-20ms`)."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.ddc import peak_pick


class DdcPeakPickTest(unittest.TestCase):
    def test_hamming_width_must_be_odd(self):
        with self.assertRaises(ValueError):
            peak_pick.hamming_smooth(np.ones(8), width=4)

    def test_find_peaks_on_impulse(self):
        salience = np.zeros(21, dtype=np.float64)
        salience[10] = 1.0
        peaks = peak_pick.find_salience_peaks(salience, threshold=0.5)
        self.assertEqual(list(peaks), [10])
        empty = peak_pick.find_salience_peaks(np.zeros(0), threshold=0.5)
        self.assertEqual(empty.size, 0)

    def test_match_onsets_within_20ms(self):
        pred = np.array([1.000, 2.000], dtype=np.float32)
        gt = np.array([1.015, 2.050], dtype=np.float32)
        counts = peak_pick.match_onsets(pred, gt, tolerance_sec=0.02)
        self.assertEqual(counts.true_positives, 1)
        self.assertEqual(counts.false_positives, 1)
        self.assertEqual(counts.false_negatives, 1)
        self.assertGreater(counts.f_score, 0.0)

    def test_match_empty_and_negative_tolerance(self):
        self.assertEqual(
            peak_pick.match_onsets(np.zeros(0), np.array([1.0])).false_negatives,
            1,
        )
        self.assertEqual(
            peak_pick.match_onsets(np.array([1.0]), np.zeros(0)).false_positives,
            1,
        )
        with self.assertRaises(ValueError):
            peak_pick.match_onsets(np.array([0.0]), np.array([0.0]), tolerance_sec=-0.1)

    def test_f_score_c_and_micro(self):
        left = peak_pick.OnsetMatchCounts(1, 0, 0)
        right = peak_pick.OnsetMatchCounts(0, 1, 1)
        pooled = peak_pick.add_counts(left, right)
        self.assertAlmostEqual(peak_pick.f_score_c([1.0, 0.0]), 0.5)
        self.assertEqual(peak_pick.f_score_c([]), 0.0)
        self.assertEqual(pooled.true_positives, 1)
        self.assertGreater(peak_pick.f_score_m(pooled), 0.0)
        self.assertEqual(peak_pick.OnsetMatchCounts(0, 0, 0).precision, 0.0)
        self.assertEqual(peak_pick.OnsetMatchCounts(0, 0, 0).recall, 0.0)
        self.assertEqual(peak_pick.OnsetMatchCounts(0, 0, 0).f_score, 0.0)

    def test_peak_times_sec(self):
        salience = np.zeros(10, dtype=np.float64)
        salience[5] = 1.0
        times = peak_pick.peak_times_sec(salience, threshold=0.2)
        self.assertAlmostEqual(float(times[0]), 0.05, places=5)


if __name__ == "__main__":
    unittest.main()
