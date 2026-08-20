"""Tests for DDC Hamming-peak snap onto ``M-slot48``."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.ddc import datasets, slot48


def _chart(
    gt_times: np.ndarray,
    *,
    bpm: float = 120.0,
    offset_sec: float = 0.0,
) -> datasets.PlacementChart:
    """Return a tiny DDC chart with BPM timing for slot snap.

    Args:
        gt_times: Ground-truth onset times in seconds.
        bpm: Constant BPM.
        offset_sec: Simfile offset.

    Returns:
        Placement chart.
    """
    times = np.asarray(gt_times, dtype=np.float64)
    return datasets.PlacementChart(
        song_key="bundle/song",
        difficulty="hard",
        spec=np.zeros((40, 80, 3), dtype=np.float32),
        target=np.zeros((40,), dtype=np.float32),
        gt_times=times,
        first_onset=0,
        last_onset=1,
        offset_sec=offset_sec,
        bpm_segments=(prep_models.BpmSegment(start_beat=0.0, bpm=bpm),),
    )


class DdcSlot48Test(unittest.TestCase):
    def test_snapping_gt_times_is_perfect(self):
        # 120 BPM: t=0 → beat 0 slot 0; t=0.25 → beat 0 slot 24.
        chart = _chart(np.array([0.0, 0.25], dtype=np.float32))
        report = slot48.evaluate_peak_times_as_slot48(
            [chart],
            [chart.gt_times],
            seed=0,
        )
        self.assertAlmostEqual(report.f1_at_05, 1.0)
        self.assertEqual(report.n_charts, 1)
        payload = slot48.report_as_dict(report)
        self.assertEqual(payload["metric"], "M-slot48")
        self.assertEqual(payload["conversion"], "ddc_hamming_peak_snap")
        self.assertNotIn("published_f1_at_05_expanded_fraxtil", payload)

    def test_empty_peaks_are_all_false_negatives(self):
        chart = _chart(np.array([0.0, 0.25], dtype=np.float32))
        report = slot48.evaluate_peak_times_as_slot48(
            [chart],
            [np.zeros((0,), dtype=np.float32)],
            seed=0,
        )
        self.assertEqual(report.f1_at_05, 0.0)
        self.assertEqual(report.counts_at_05.true_positives, 0)
        self.assertEqual(report.counts_at_05.false_positives, 0)
        self.assertEqual(report.counts_at_05.false_negatives, 2)

    def test_extra_peak_is_a_false_positive(self):
        chart = _chart(np.array([0.0], dtype=np.float64))
        pred = np.array([0.0, 0.25], dtype=np.float64)
        report = slot48.evaluate_peak_times_as_slot48([chart], [pred], seed=0)
        self.assertEqual(report.counts_at_05.true_positives, 1)
        self.assertEqual(report.counts_at_05.false_positives, 1)
        self.assertEqual(report.counts_at_05.false_negatives, 0)

    def test_late_peak_does_not_extend_gt_beats(self):
        # 120 BPM: GT at t=0 is beat 0; t=1.0 is beat 2 and must not pad GT.
        chart = _chart(np.array([0.0], dtype=np.float64))
        pred = np.array([0.0, 1.0], dtype=np.float64)
        report = slot48.evaluate_peak_times_as_slot48([chart], [pred], seed=0)
        self.assertEqual(report.n_beats, 1)
        self.assertEqual(report.counts_at_05.true_positives, 1)
        self.assertEqual(report.counts_at_05.false_positives, 1)
        self.assertEqual(report.counts_at_05.false_negatives, 0)

    def test_missing_bpm_raises(self):
        chart = _chart(np.array([0.0], dtype=np.float32))
        chart.bpm_segments = ()
        with self.assertRaises(ValueError):
            slot48.evaluate_peak_times_as_slot48(
                [chart],
                [chart.gt_times],
                seed=0,
            )

    def test_empty_charts_raise(self):
        with self.assertRaises(ValueError):
            slot48.evaluate_peak_times_as_slot48([], [], seed=0)
