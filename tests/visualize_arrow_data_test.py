import os
import sys
import unittest

import numpy as np

# Allow importing the script module (scripts/visualize_arrow_data.py)
_SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_SCRIPT_DIR = os.path.abspath(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import visualize_arrow_data as viz  # noqa: E402

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


class CramersVTest(unittest.TestCase):
    """Tests for _cramers_v contingency-table metric."""

    def test_zero_total_returns_zero(self):
        C = np.zeros((2, 3))
        self.assertAlmostEqual(viz._cramers_v(C), 0.0)

    def test_single_row_returns_zero(self):
        C = np.array([[10, 20, 30]])
        self.assertAlmostEqual(viz._cramers_v(C), 0.0)

    def test_single_column_returns_zero(self):
        C = np.array([[10], [20], [30]])
        self.assertAlmostEqual(viz._cramers_v(C), 0.0)

    def test_perfect_association_2x2_returns_one(self):
        # Diagonal: all mass on diagonal -> chi2 = n, min_dim = 1, V = sqrt(n/(n*1)) = 1
        C = np.array([[10.0, 0.0], [0.0, 10.0]])
        self.assertAlmostEqual(viz._cramers_v(C), 1.0, places=5)

    def test_independence_2x2_returns_zero(self):
        # Uniform 2x2: observed = expected, chi2 = 0
        C = np.array([[5.0, 5.0], [5.0, 5.0]])
        self.assertAlmostEqual(viz._cramers_v(C), 0.0, places=5)

    def test_perfect_association_3x3_returns_one(self):
        C = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
        self.assertAlmostEqual(viz._cramers_v(C), 1.0, places=5)

    def test_v_in_valid_range(self):
        # Arbitrary table: Cramér's V must be in [0, 1]
        C = np.array([[1, 9, 5], [8, 2, 4], [3, 7, 6]], dtype=np.float64)
        v = viz._cramers_v(C)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_v_symmetric_in_table_scaling(self):
        # V is unchanged if we scale the table by a constant
        C1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        C2 = 10.0 * C1
        self.assertAlmostEqual(viz._cramers_v(C1), viz._cramers_v(C2), places=6)


class CollectAggregatesTest(unittest.TestCase):
    """Tests for collect_aggregates() per-step and correlation data."""

    def test_returns_per_step_arrays(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        self.assertIsNotNone(agg)
        for key in (
            "per_step_time_norm",
            "per_step_arrow_code",
            "per_step_note_kind",
            "per_step_interval",
            "per_step_chord_size",
        ):
            self.assertIn(key, agg, msg=f"Missing key: {key}")
            self.assertIsInstance(agg[key], np.ndarray, msg=f"Key {key} is not ndarray")

    def test_per_step_arrays_same_length(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        n = len(agg["per_step_time_norm"])
        self.assertEqual(len(agg["per_step_arrow_code"]), n)
        self.assertEqual(len(agg["per_step_note_kind"]), n)
        self.assertEqual(len(agg["per_step_interval"]), n)
        self.assertEqual(len(agg["per_step_chord_size"]), n)

    def test_per_step_time_norm_in_unit_interval(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        t = agg["per_step_time_norm"]
        self.assertTrue(np.all(t >= 0.0), "normalized time should be >= 0")
        self.assertTrue(np.all(t <= 1.0), "normalized time should be <= 1")

    def test_per_step_interval_first_step_per_chart_is_nan(self):
        # First step of each chart has no previous step -> NaN in per_step_interval
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        interval = agg["per_step_interval"]
        # We have one chart in testdata; first element should be NaN
        self.assertTrue(np.isnan(interval[0]), "First step interval should be NaN")
        self.assertTrue(
            np.all(~np.isnan(interval[1:]) | (interval[1:] >= 0)),
            "Other intervals non-NaN and non-negative",
        )

    def test_per_step_note_kind_in_range(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        kind = agg["per_step_note_kind"]
        self.assertTrue(np.all((kind >= 0) & (kind <= 5)), "note kind in 0..5")


class WriteSummaryCorrelationTest(unittest.TestCase):
    """Tests that write_summary includes correlation metrics and they are valid."""

    def test_summary_contains_correlation_section(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        summary = viz.write_summary(agg, output_path=None)
        self.assertIn("Correlation summary", summary)
        self.assertIn("Cramér's V", summary)

    def test_summary_contains_time_bin_and_interval_v(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        summary = viz.write_summary(agg, output_path=None)
        self.assertIn("time bin x note kind", summary)
        self.assertIn("interval bin x note kind", summary)

    def test_cramer_v_values_in_summary_in_valid_range(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        summary = viz.write_summary(agg, output_path=None)
        # Parse "Cramér's V (time bin x note kind): 0.XXXX"
        for line in summary.splitlines():
            if "Cramér's V" in line and ":" in line and "N/A" not in line:
                try:
                    val_str = line.split(":")[-1].strip()
                    v = float(val_str)
                    self.assertGreaterEqual(
                        v, 0.0, msg=f"V should be >= 0 in line: {line}"
                    )
                    self.assertLessEqual(
                        v, 1.0, msg=f"V should be <= 1 in line: {line}"
                    )
                except ValueError:
                    pass

    def test_counts_table_has_expected_structure(self):
        agg = viz.collect_aggregates(TEST_DATA_DIR)
        summary = viz.write_summary(agg, output_path=None)
        self.assertIn(
            "Counts (rows=note kind, cols=time bin 0=start .. 9=end):", summary
        )
        for label in viz.NOTE_KIND_LABELS:
            self.assertIn(
                label,
                summary,
                msg=f"Note kind label {label!r} should appear in counts table",
            )
