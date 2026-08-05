"""Tests for AR free-run gate metrics (primary vs target_times, aux vs chart)."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest

import numpy as np

os.environ.setdefault("STEPCOVNET_NO_WSL", "1")

REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eval_ar_onset_offline as debug  # noqa: E402


class ArDecodeGateMetricsTest(unittest.TestCase):
    def test_primary_vs_target_aux_vs_chart(self) -> None:
        target = np.array([0.0, 0.02, 0.04], dtype=np.float32)
        chart = np.array([0.0, 0.05, 0.04], dtype=np.float32)
        pred = np.array([0.0, 0.021, 0.04], dtype=np.float32)

        report = debug._ar_decode_gate_metrics(
            pred,
            target,
            chart,
            tolerance_sec=0.02,
            ar_decode_length=3,
            stopped_on_eos=False,
        )

        primary = report["ordered_onset_match"]
        chart_block = report["chart_ordered_onset_match"]
        assert isinstance(primary, dict)
        assert isinstance(chart_block, dict)

        self.assertEqual(int(primary["n_matched"]), 3)
        self.assertEqual(float(primary["rate"]), 1.0)
        self.assertEqual(int(chart_block["n_matched"]), 2)
        self.assertLess(float(chart_block["rate"]), 1.0)

    def test_onset_reference_times_order(self) -> None:
        batch = {
            "onset_step_mask": np.array([[1.0, 1.0, 0.0, 1.0]]),
            "target_times": np.array([[0.1, 0.2, 0.0, 0.4]]),
            "gt_mask": np.array([[1.0, 1.0, 1.0]]),
            "gt_times": np.array([[0.11, 0.21, 0.31]]),
        }
        target, chart = debug._onset_reference_times(batch)
        np.testing.assert_allclose(target, [0.1, 0.2, 0.4])
        np.testing.assert_allclose(chart, [0.11, 0.21, 0.31])


class TeacherPreflightGateTest(unittest.TestCase):
    def test_overfit_requires_perfect_teacher(self) -> None:
        report = {
            "ordered_onset_match": {
                "n_matched": 633,
                "n_denom": 634,
                "rate": 633 / 634,
            },
        }
        passed, reason = debug._teacher_gate_passes(report, split="overfit")
        self.assertFalse(passed)
        self.assertIn("not perfect", reason)

        report["ordered_onset_match"] = {
            "n_matched": 634,
            "n_denom": 634,
            "rate": 1.0,
        }
        passed, _ = debug._teacher_gate_passes(report, split="overfit")
        self.assertTrue(passed)

    def test_val_skips_when_timing_and_null_skill_are_bad(self) -> None:
        report = {
            "ordered_onset_match": {
                "n_matched": 0,
                "n_denom": 380,
                "rate": 0.0,
            },
            "null_baseline": {
                "skill_timing_match": -0.11,
                "skill_event_f1": -0.08,
            },
        }
        passed, reason = debug._teacher_gate_passes(report, split="val")
        self.assertFalse(passed)
        self.assertIn("below min", reason)

    def test_val_passes_on_positive_null_skill(self) -> None:
        report = {
            "ordered_onset_match": {
                "n_matched": 1,
                "n_denom": 380,
                "rate": 1 / 380,
            },
            "null_baseline": {
                "skill_timing_match": 0.02,
                "skill_event_f1": -0.08,
            },
        }
        passed, reason = debug._teacher_gate_passes(report, split="val")
        self.assertTrue(passed)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
