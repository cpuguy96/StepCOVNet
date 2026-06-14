import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_events import matching
from stepcovnet.onset_events import metrics


class MetricsTest(unittest.TestCase):
    def test_default_constants(self):
        self.assertEqual(metrics.DEFAULT_TOLERANCE_SEC, 0.02)
        self.assertEqual(metrics.DEFAULT_CONFIDENCE_THRESHOLD, 0.5)
        self.assertEqual(metrics.DEFAULT_MIN_ONSET_DISTANCE_MS, 50.0)

    def test_min_gap_metric_collapses_clustered_predictions(self):
        pred_times = np.array([[1.0, 1.01, 1.02, 2.0]], dtype=np.float32)
        pred_confidence = np.array([[0.9, 0.9, 0.9, 0.9]], dtype=np.float32)
        gt_times = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

        tp_raw, fp_raw, fn_raw = metrics.count_event_onset_errors_numpy(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
            confidence_threshold=0.5,
        )
        self.assertEqual(tp_raw, 2)
        self.assertEqual(fp_raw, 2)
        self.assertEqual(fn_raw, 0)

        tp_gap, fp_gap, fn_gap = metrics.count_event_onset_errors_numpy(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
            confidence_threshold=0.5,
            min_onset_distance_ms=50.0,
        )
        self.assertEqual(tp_gap, 2)
        self.assertEqual(fp_gap, 0)
        self.assertEqual(fn_gap, 0)

    def test_filter_predicted_onsets_matches_inference_steps(self):
        times = np.array([2.0, 0.5, 0.52, 5.0], dtype=np.float32)
        conf = np.array([0.9, 0.9, 0.8, 0.2], dtype=np.float32)
        filtered_times, filtered_conf = metrics.filter_predicted_onsets_numpy(
            times,
            conf,
            confidence_threshold=0.5,
            min_onset_distance_ms=50.0,
        )
        np.testing.assert_allclose(
            filtered_times, np.array([0.5, 2.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            filtered_conf, np.array([0.9, 0.9], dtype=np.float32)
        )

    def test_invalid_min_onset_distance_raises(self):
        pred_times = np.array([1.0], dtype=np.float32)
        pred_confidence = np.array([0.9], dtype=np.float32)
        gt_times = np.array([1.0], dtype=np.float32)
        gt_mask = np.array([1.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            metrics.count_event_onset_errors_numpy(
                pred_times,
                pred_confidence,
                gt_times,
                gt_mask,
                tolerance_sec=0.02,
                confidence_threshold=0.5,
                min_onset_distance_ms=-1.0,
            )

    def test_perfect_match(self):
        pred_times = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        pred_confidence = np.array([0.9, 0.8, 0.95], dtype=np.float32)
        gt_times = np.array([0.51, 1.01, 2.0, 0.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_confidence_filters_false_positives(self):
        pred_times = np.array([0.5, 5.0], dtype=np.float32)
        pred_confidence = np.array([0.9, 0.2], dtype=np.float32)
        gt_times = np.array([0.51, 0.0], dtype=np.float32)
        gt_mask = np.array([1.0, 0.0], dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_false_positive_and_false_negative(self):
        pred_times = np.array([1.0, 5.0], dtype=np.float32)
        pred_confidence = np.array([0.9, 0.9], dtype=np.float32)
        gt_times = np.array([1.0, 2.0, 0.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0, 0.0], dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_no_high_confidence_predictions(self):
        pred_times = np.array([1.0, 2.0], dtype=np.float32)
        pred_confidence = np.array([0.1, 0.2], dtype=np.float32)
        gt_times = np.array([1.0, 2.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0], dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertAlmostEqual(precision, 0.0)
        self.assertAlmostEqual(recall, 0.0)
        self.assertAlmostEqual(f1, 0.0)

    def test_no_valid_ground_truth(self):
        pred_times = np.array([1.0, 2.0], dtype=np.float32)
        pred_confidence = np.array([0.9, 0.8], dtype=np.float32)
        gt_times = np.zeros(2, dtype=np.float32)
        gt_mask = np.zeros(2, dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertAlmostEqual(precision, 0.0)
        self.assertAlmostEqual(recall, 0.0)
        self.assertAlmostEqual(f1, 0.0)

    def test_empty_inputs(self):
        pred_times = np.array([], dtype=np.float32)
        pred_confidence = np.array([], dtype=np.float32)
        gt_times = np.array([], dtype=np.float32)
        gt_mask = np.array([], dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertAlmostEqual(precision, 0.0)
        self.assertAlmostEqual(recall, 0.0)
        self.assertAlmostEqual(f1, 0.0)

    def test_tolerance_boundary(self):
        pred_times = np.array([0.0, 0.02], dtype=np.float32)
        pred_confidence = np.array([0.9, 0.9], dtype=np.float32)
        gt_times = np.array([0.02, 0.05], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0], dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
        )

        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_batch_aggregation(self):
        pred_times = np.array(
            [[0.5, 5.0], [10.0, 11.0]],
            dtype=np.float32,
        )
        pred_confidence = np.array(
            [[0.9, 0.9], [0.8, 0.8]],
            dtype=np.float32,
        )
        gt_times = np.array(
            [[0.51, 0.0], [10.01, 11.01]],
            dtype=np.float32,
        )
        gt_mask = np.array(
            [[1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )

        precision, recall, f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 6.0 / 7.0)

    def test_public_api_numpy_returns_arrays(self):
        pred_times = np.array([1.0], dtype=np.float32)
        pred_confidence = np.array([0.9], dtype=np.float32)
        gt_times = np.array([1.0], dtype=np.float32)
        gt_mask = np.array([1.0], dtype=np.float32)

        precision, recall, f1 = metrics.event_onset_f1(
            pred_times, pred_confidence, gt_times, gt_mask
        )

        self.assertIsInstance(precision, np.ndarray)
        self.assertAlmostEqual(float(precision), 1.0)
        self.assertAlmostEqual(float(recall), 1.0)
        self.assertAlmostEqual(float(f1), 1.0)

    def test_event_onset_f1_tensorflow_matches_numpy(self):
        pred_times = np.array([[0.5, 1.0, 5.0]], dtype=np.float32)
        pred_confidence = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
        gt_times = np.array([[0.51, 1.01, 0.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

        numpy_precision, numpy_recall, numpy_f1 = metrics.event_onset_f1_numpy(
            pred_times, pred_confidence, gt_times, gt_mask
        )
        tf_precision, tf_recall, tf_f1 = metrics.event_onset_f1(
            tf.constant(pred_times),
            tf.constant(pred_confidence),
            tf.constant(gt_times),
            tf.constant(gt_mask),
        )

        self.assertAlmostEqual(float(tf_precision.numpy()), numpy_precision)
        self.assertAlmostEqual(float(tf_recall.numpy()), numpy_recall)
        self.assertAlmostEqual(float(tf_f1.numpy()), numpy_f1)

    def test_invalid_tolerance_raises(self):
        pred_times = np.array([1.0], dtype=np.float32)
        pred_confidence = np.array([0.9], dtype=np.float32)
        gt_times = np.array([1.0], dtype=np.float32)
        gt_mask = np.array([1.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            metrics.event_onset_f1_numpy(
                pred_times,
                pred_confidence,
                gt_times,
                gt_mask,
                tolerance_sec=-0.01,
            )

    def test_invalid_confidence_threshold_raises(self):
        pred_times = np.array([1.0], dtype=np.float32)
        pred_confidence = np.array([0.9], dtype=np.float32)
        gt_times = np.array([1.0], dtype=np.float32)
        gt_mask = np.array([1.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            metrics.event_onset_f1_numpy(
                pred_times,
                pred_confidence,
                gt_times,
                gt_mask,
                confidence_threshold=1.5,
            )

    def test_pred_confidence_shape_mismatch_raises(self):
        pred_times = np.array([1.0, 2.0], dtype=np.float32)
        pred_confidence = np.array([0.9], dtype=np.float32)
        gt_times = np.array([1.0, 2.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            metrics.event_onset_f1_numpy(pred_times, pred_confidence, gt_times, gt_mask)

    def test_batch_shape_mismatch_raises(self):
        pred_times = np.array([[1.0, 2.0]], dtype=np.float32)
        pred_confidence = np.array([[0.9, 0.8]], dtype=np.float32)
        gt_times = np.array([[1.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 0.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            metrics.event_onset_f1_numpy(pred_times, pred_confidence, gt_times, gt_mask)

    def test_count_event_onset_errors_all_k_slots(self):
        pred_times = np.array([[0.05, 0.10, 0.15, 50.0, 51.0, 52.0]], dtype=np.float32)
        pred_confidence = np.array(
            [[0.9, 0.9, 0.9, 0.95, 0.95, 0.95]], dtype=np.float32
        )
        gt_times = np.array([[0.05, 0.10, 0.15, 0.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)

        tp, fp, fn = metrics.count_event_onset_errors_numpy(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
            confidence_threshold=0.5,
        )

        self.assertEqual(tp, 3)
        self.assertEqual(fp, 3)
        self.assertEqual(fn, 0)

    def test_prefilter_regression_junk_only_passes_threshold(self):
        """Pre-filter matching saw only far-away junk; all-K still scores aligned slots."""
        pred_times = np.array([[0.05, 0.10, 0.15, 50.0, 51.0, 52.0]], dtype=np.float32)
        pred_confidence = np.array([[0.4, 0.4, 0.4, 0.9, 0.9, 0.9]], dtype=np.float32)
        gt_times = np.array([[0.05, 0.10, 0.15, 0.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)

        keep = pred_confidence[0] >= 0.5
        prefilter_result = matching.match_onsets_numpy(
            pred_times[0][keep],
            gt_times[0:1],
            gt_mask[0:1],
            tolerance_sec=0.02,
        )
        self.assertEqual(int(prefilter_result.num_matches[0]), 0)

        tp, fp, fn = metrics.count_event_onset_errors_numpy(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
            confidence_threshold=0.5,
        )
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 3)
        self.assertEqual(fn, 3)

        pred_confidence_high = pred_confidence.copy()
        pred_confidence_high[0, :3] = 0.6
        tp_fixed, fp_fixed, fn_fixed = metrics.count_event_onset_errors_numpy(
            pred_times,
            pred_confidence_high,
            gt_times,
            gt_mask,
            tolerance_sec=0.02,
            confidence_threshold=0.5,
        )
        self.assertEqual(tp_fixed, 3)
        self.assertEqual(fp_fixed, 3)
        self.assertEqual(fn_fixed, 0)
        _precision, _recall, f1 = metrics.event_onset_f1_numpy(
            pred_times,
            pred_confidence_high,
            gt_times,
            gt_mask,
        )
        self.assertAlmostEqual(f1, 2.0 / 3.0)

    def test_invalid_rank_raises(self):
        pred_times = np.array([[[1.0]]], dtype=np.float32)
        pred_confidence = np.array([[[0.9]]], dtype=np.float32)
        gt_times = np.array([[1.0]], dtype=np.float32)
        gt_mask = np.array([[1.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            metrics.event_onset_f1_numpy(pred_times, pred_confidence, gt_times, gt_mask)


if __name__ == "__main__":
    unittest.main()
