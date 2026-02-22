import unittest

import numpy as np

from stepcovnet import metrics


class OnsetF1MetricTest(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.OnsetF1Metric(tolerance=1, threshold=0.5)

    def test_update_state_perfect_match(self):
        y_true = np.array([[0, 1, 0, 0, 1]])
        y_pred = np.array([[0, 1, 0, 0, 1]])
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 1.0)

    def test_update_state_within_tolerance(self):
        y_true = np.array([[0, 1, 0, 0, 0]])
        y_pred = np.array([[0, 0, 1, 0, 0]])  # Shifted by 1, within tolerance
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 1.0, places=6)

    def test_update_state_outside_tolerance(self):
        y_true = np.array([[0, 1, 0, 0, 0]])
        y_pred = np.array([[0, 0, 0, 1, 0]])  # Shifted by 2, outside tolerance
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 0.0)

    def test_update_state_false_positive(self):
        y_true = np.array([[0, 0, 0, 0, 0]])
        y_pred = np.array([[0, 1, 0, 0, 0]])
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 0.0)

    def test_update_state_false_negative(self):
        y_true = np.array([[0, 1, 0, 0, 0]])
        y_pred = np.array([[0, 0, 0, 0, 0]])
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 0.0)

    def test_update_state_rank_2_input(self):
        # Shape (batch, time)
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0, 1, 0]])
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 1.0, places=6)

    def test_update_state_rank_3_input(self):
        # Shape (batch, time, 1)
        y_true = np.array([[[0], [1], [0]]])
        y_pred = np.array([[[0], [1], [0]]])
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 1.0, places=6)

    def test_update_state_rank_4_input(self):
        # Shape (batch, time, 1, 1)
        y_true = np.array([[[[0]], [[1]], [[0]]]])
        y_pred = np.array([[[[0]], [[1]], [[0]]]])
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 1.0, places=6)

    def test_reset_state(self):
        y_true = np.array([[0, 1, 0]])
        y_pred = np.array([[0, 1, 0]])
        self.metric.update_state(y_true, y_pred)
        self.metric.reset_state()
        result = self.metric.result()
        self.assertAlmostEqual(result.numpy(), 0.0)

    def test_get_config(self):
        config = self.metric.get_config()
        self.assertEqual(config["tolerance"], 1)
        self.assertEqual(config["threshold"], 0.5)
        self.assertEqual(config["name"], "onset_f1_score")


def _one_hot_pred(batch_size, seq_len, num_classes, class_indices):
    """Build y_pred so argmax gives class_indices. class_indices shape (batch, seq)."""
    y = np.zeros((batch_size, seq_len, num_classes), dtype=np.float32)
    for b in range(batch_size):
        for s in range(seq_len):
            y[b, s, class_indices[b, s]] = 1.0
    return y


class ArrowDistributionMatchMetricTest(unittest.TestCase):
    def setUp(self):
        self.metric = metrics.ArrowDistributionMatchMetric(
            num_classes=4, ignore_class=0, name="arrow_dist_match"
        )

    def test_perfect_match_returns_one(self):
        # Predictions exactly match labels -> same distribution -> JSD=0 -> result=1
        y_true = np.array([[1, 2, 3]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 3, 4, np.array([[1, 2, 3]]))
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_total_mismatch_returns_low(self):
        # All true are class 1, all pred are class 2 -> distributions differ
        y_true = np.array([[1, 1, 1]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 3, 4, np.array([[2, 2, 2]]))
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertLess(float(result), 0.5)
        self.assertGreaterEqual(float(result), 0.0)

    def test_ignore_class_zero_excludes_padding(self):
        # Positions with y_true=0 are not counted; pred matches true on non-padding
        y_true = np.array([[0, 1, 0, 2]], dtype=np.int32)
        class_indices = np.array([[0, 1, 0, 2]])  # pred at 0,2 are ignored
        y_pred = _one_hot_pred(1, 4, 4, class_indices)
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_reset_state_clears_accumulation(self):
        y_true = np.array([[1, 2]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 2, 4, np.array([[1, 2]]))
        self.metric.update_state(y_true, y_pred)
        self.metric.reset_state()
        # After reset, update with same data again; result should be 1.0 (not 2x counts)
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_get_config(self):
        m = metrics.ArrowDistributionMatchMetric(
            num_classes=8, ignore_class=0, name="custom_name"
        )
        config = m.get_config()
        self.assertEqual(config["num_classes"], 8)
        self.assertEqual(config["ignore_class"], 0)
        self.assertEqual(config["name"], "custom_name")

    def test_batch_and_sequence_aggregation(self):
        # Multiple batch items and steps; pred matches true -> result 1
        y_true = np.array([[1, 2], [3, 1]], dtype=np.int32)
        y_pred = _one_hot_pred(2, 2, 4, np.array([[1, 2], [3, 1]]))
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_default_256_classes_accepts_arrow_shapes(self):
        # Smoke test: metric with default num_classes=256 and (1, 10, 256) pred
        metric_256 = metrics.ArrowDistributionMatchMetric()
        y_true = np.ones((1, 10), dtype=np.int32)  # all class 1
        y_pred = np.zeros((1, 10, 256), dtype=np.float32)
        y_pred[:, :, 1] = 1.0
        metric_256.update_state(y_true, y_pred)
        result = metric_256.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)


class ArrowNoteKindDistributionMetricTest(unittest.TestCase):
    """Arrow codes: 0=empty, 1=0001 single, 5=0011 chord, 2=0002 hold_start, 3=0003 hold_end, 11=0023 hold_both."""

    def setUp(self):
        self.metric = metrics.ArrowNoteKindDistributionMetric(
            num_note_kinds=6, ignore_class=0, name="arrow_note_kind_dist_match"
        )

    def test_perfect_match_returns_one(self):
        y_true = np.array([[1, 5, 2]], dtype=np.int32)  # single, chord, hold_start
        y_pred = _one_hot_pred(1, 3, 256, np.array([[1, 5, 2]]))
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_mismatch_returns_lower(self):
        y_true = np.array([[1, 1, 1]], dtype=np.int32)  # all single
        y_pred = _one_hot_pred(1, 3, 256, np.array([[5, 5, 5]]))  # all chord
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertLess(float(result), 1.0)
        self.assertGreaterEqual(float(result), 0.0)

    def test_ignore_class_zero_excludes_padding(self):
        y_true = np.array([[0, 1, 0, 2]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 4, 256, np.array([[0, 1, 0, 2]]))
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_reset_state(self):
        y_true = np.array([[1, 2]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 2, 256, np.array([[1, 2]]))
        self.metric.update_state(y_true, y_pred)
        self.metric.reset_state()
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_get_config(self):
        m = metrics.ArrowNoteKindDistributionMetric(
            num_note_kinds=6, ignore_class=0, name="custom"
        )
        config = m.get_config()
        self.assertEqual(config["num_note_kinds"], 6)
        self.assertEqual(config["ignore_class"], 0)
        self.assertEqual(config["name"], "custom")


class ArrowHoldValidityMetricTest(unittest.TestCase):
    """Hold rules: 3 must follow 2 in same column; 3 cannot follow 1 in same column."""

    def setUp(self):
        self.metric = metrics.ArrowHoldValidityMetric(
            ignore_class=0, name="arrow_hold_validity"
        )

    def test_valid_hold_sequence_returns_one(self):
        # Column 0: 2 then 3. Codes: 2=0002 (hold start col0), 3=0003 (hold end col0)
        # So sequence [2, 3] is valid. y_true for mask (no padding).
        y_true = np.array([[1, 1]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 2, 256, np.array([[2, 3]]))
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_unmatched_hold_end_is_violation(self):
        # 3 with no preceding 2 in that column -> violation
        y_true = np.array([[1, 1]], dtype=np.int32)
        y_pred = _one_hot_pred(
            1, 2, 256, np.array([[3, 1]])
        )  # 3 at first step (no 2 before)
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertLess(float(result), 1.0)

    def test_hold_end_after_tap_is_violation(self):
        # 1 then 3 in same column -> violation
        y_true = np.array([[1, 1]], dtype=np.int32)
        y_pred = _one_hot_pred(
            1, 2, 256, np.array([[1, 3]])
        )  # tap then hold end in col0
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertLess(float(result), 1.0)

    def test_no_hold_ends_returns_one(self):
        y_true = np.array([[1, 1]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 2, 256, np.array([[1, 1]]))  # no 2 or 3
        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_reset_state(self):
        y_true = np.array([[1]], dtype=np.int32)
        y_pred = _one_hot_pred(1, 1, 256, np.array([[3]]))
        self.metric.update_state(y_true, y_pred)
        self.metric.reset_state()
        y_pred_valid = _one_hot_pred(1, 2, 256, np.array([[2, 3]]))
        y_true_valid = np.array([[1, 1]], dtype=np.int32)
        self.metric.update_state(y_true_valid, y_pred_valid)
        result = self.metric.result()
        self.assertAlmostEqual(float(result), 1.0, places=5)

    def test_get_config(self):
        m = metrics.ArrowHoldValidityMetric(ignore_class=0, name="custom")
        config = m.get_config()
        self.assertEqual(config["ignore_class"], 0)
        self.assertEqual(config["name"], "custom")


class ComputeHoldValidityViolationsTest(unittest.TestCase):
    """Tests for compute_hold_validity_violations (numpy helper used by check script)."""

    def test_valid_hold_sequence_zero_violations(self):
        # 2 then 3 in column 0 -> valid
        codes = np.array([2, 3], dtype=np.int32)
        violations, hold_ends, examples = metrics.compute_hold_validity_violations(
            codes
        )
        self.assertEqual(violations, 0)
        self.assertEqual(hold_ends, 1)
        self.assertEqual(examples, [])

    def test_unmatched_hold_end_violation(self):
        # 3 with no preceding 2
        codes = np.array([3, 1], dtype=np.int32)
        violations, hold_ends, examples = metrics.compute_hold_validity_violations(
            codes
        )
        self.assertGreater(violations, 0)
        self.assertEqual(hold_ends, 1)
        # Code 3 = "0003" -> digit 3 in column 3
        self.assertIn((0, 3, "unmatched_3"), examples)

    def test_hold_end_after_tap_violation(self):
        # 1 then 3 in same column
        codes = np.array([1, 3], dtype=np.int32)
        violations, hold_ends, examples = metrics.compute_hold_validity_violations(
            codes
        )
        self.assertGreater(violations, 0)
        self.assertEqual(hold_ends, 1)
        # Code 1 = "0001", code 3 = "0003" -> column 3 has 1 then 3
        self.assertIn((1, 3, "3_after_1"), examples)

    def test_no_hold_ends_zero_hold_ends(self):
        codes = np.array([1, 1, 0], dtype=np.int32)
        violations, hold_ends, examples = metrics.compute_hold_validity_violations(
            codes
        )
        self.assertEqual(violations, 0)
        self.assertEqual(hold_ends, 0)
        self.assertEqual(examples, [])


if __name__ == "__main__":
    unittest.main()
