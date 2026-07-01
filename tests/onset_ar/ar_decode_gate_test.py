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

import debug_ar_onset_overfit as debug  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
