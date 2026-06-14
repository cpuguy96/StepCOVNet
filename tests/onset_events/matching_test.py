import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_events import matching


class MatchingTest(unittest.TestCase):
    def test_default_tolerance_constant(self):
        self.assertEqual(matching.DEFAULT_TOLERANCE_SEC, 0.02)

    def test_perfect_one_to_one_match(self):
        pred_times = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        gt_times = np.array([0.51, 1.01, 2.0, 0.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)

        result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

        self.assertEqual(int(result.num_matches[0]), 3)
        np.testing.assert_array_equal(result.matched_pred_indices[0, :3], [0, 1, 2])
        np.testing.assert_array_equal(result.matched_gt_indices[0, :3], [0, 1, 2])
        self.assertFalse(result.pred_unmatched_mask.any())
        self.assertFalse(result.gt_unmatched_mask[0, :3].any())
        self.assertFalse(result.gt_unmatched_mask[0, 3])

    def test_no_valid_gt_onsets(self):
        pred_times = np.array([0.5, 1.0], dtype=np.float32)
        gt_times = np.zeros(4, dtype=np.float32)
        gt_mask = np.zeros(4, dtype=np.float32)

        result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

        self.assertEqual(int(result.num_matches[0]), 0)
        np.testing.assert_array_equal(result.matched_pred_indices[0], [-1, -1])
        self.assertTrue(result.pred_unmatched_mask.all())
        self.assertFalse(result.gt_unmatched_mask.any())

    def test_no_pred_within_tolerance(self):
        pred_times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        gt_times = np.array([2.0, 3.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0], dtype=np.float32)

        result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

        self.assertEqual(int(result.num_matches[0]), 0)
        self.assertTrue(result.pred_unmatched_mask.all())
        self.assertTrue(result.gt_unmatched_mask[0, :2].all())

    def test_tolerance_boundary(self):
        pred_times = np.array([0.0, 0.02], dtype=np.float32)
        gt_times = np.array([0.02, 0.05], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0], dtype=np.float32)

        result = matching.match_onsets_numpy(
            pred_times, gt_times, gt_mask, tolerance_sec=0.02
        )

        self.assertEqual(int(result.num_matches[0]), 1)
        self.assertEqual(int(result.matched_pred_indices[0, 0]), 1)
        self.assertEqual(int(result.matched_gt_indices[0, 0]), 0)
        self.assertTrue(result.pred_unmatched_mask[0, 0])
        self.assertTrue(result.gt_unmatched_mask[0, 1])

    def test_more_predictions_than_gt(self):
        pred_times = np.array([1.0, 1.01, 1.02, 5.0], dtype=np.float32)
        gt_times = np.array([1.005, 0.0], dtype=np.float32)
        gt_mask = np.array([1.0, 0.0], dtype=np.float32)

        result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

        self.assertEqual(int(result.num_matches[0]), 1)
        self.assertEqual(int(result.matched_gt_indices[0, 0]), 0)
        self.assertEqual(int(np.sum(~result.pred_unmatched_mask[0])), 1)
        self.assertFalse(result.gt_unmatched_mask[0, 0])

    def test_more_gt_than_predictions(self):
        pred_times = np.array([1.0, 3.0], dtype=np.float32)
        gt_times = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

        self.assertEqual(int(result.num_matches[0]), 2)
        self.assertTrue(result.gt_unmatched_mask[0, 1])
        self.assertFalse(result.gt_unmatched_mask[0, [0, 2]].any())

    def test_competing_predictions_pick_closest(self):
        pred_times = np.array([1.0, 1.005, 2.0], dtype=np.float32)
        gt_times = np.array([1.002, 2.0], dtype=np.float32)
        gt_mask = np.array([1.0, 1.0], dtype=np.float32)

        result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

        self.assertEqual(int(result.num_matches[0]), 2)
        matched_preds = set(result.matched_pred_indices[0, :2].tolist())
        self.assertIn(0, matched_preds)
        self.assertIn(2, matched_preds)
        self.assertNotIn(1, matched_preds)

    def test_batch_dimension(self):
        pred_times = np.array(
            [[0.5, 1.0], [10.0, 11.0]],
            dtype=np.float32,
        )
        gt_times = np.array(
            [[0.51, 1.01, 0.0], [10.01, 11.01, 0.0]],
            dtype=np.float32,
        )
        gt_mask = np.array(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )

        result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

        self.assertEqual(result.num_matches.shape, (2,))
        self.assertEqual(int(result.num_matches[0]), 2)
        self.assertEqual(int(result.num_matches[1]), 2)
        self.assertFalse(result.pred_unmatched_mask.any())

    def test_optional_pred_confidence_accepted(self):
        pred_times = np.array([1.0], dtype=np.float32)
        gt_times = np.array([1.0], dtype=np.float32)
        gt_mask = np.array([1.0], dtype=np.float32)
        pred_confidence = np.array([0.9], dtype=np.float32)

        result = matching.match_onsets_numpy(
            pred_times, gt_times, gt_mask, pred_confidence=pred_confidence
        )

        self.assertEqual(int(result.num_matches[0]), 1)

    def test_invalid_tolerance_raises(self):
        pred_times = np.array([1.0], dtype=np.float32)
        gt_times = np.array([1.0], dtype=np.float32)
        gt_mask = np.array([1.0], dtype=np.float32)

        with self.assertRaises(ValueError):
            matching.match_onsets_numpy(
                pred_times, gt_times, gt_mask, tolerance_sec=-0.01
            )

    def test_shape_mismatch_raises(self):
        pred_times = np.array([[1.0, 2.0]], dtype=np.float32)
        gt_times = np.array([[1.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 0.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

    def test_invalid_rank_raises(self):
        pred_times = np.array([[[1.0]]], dtype=np.float32)
        gt_times = np.array([[1.0]], dtype=np.float32)
        gt_mask = np.array([[1.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            matching.match_onsets_numpy(pred_times, gt_times, gt_mask)

    def test_match_onsets_tensorflow_matches_numpy(self):
        pred_times = np.array([[0.5, 1.0, 5.0]], dtype=np.float32)
        gt_times = np.array([[0.51, 1.01, 0.0]], dtype=np.float32)
        gt_mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)

        numpy_result = matching.match_onsets_numpy(pred_times, gt_times, gt_mask)
        tf_result = matching.match_onsets(
            tf.constant(pred_times),
            tf.constant(gt_times),
            tf.constant(gt_mask),
        )

        matched_pred, matched_gt, num_matches, pred_unmatched, gt_unmatched = tf_result
        self.assertEqual(int(num_matches.numpy()[0]), int(numpy_result.num_matches[0]))
        np.testing.assert_array_equal(
            matched_pred.numpy(), numpy_result.matched_pred_indices
        )
        np.testing.assert_array_equal(
            matched_gt.numpy(), numpy_result.matched_gt_indices
        )
        np.testing.assert_array_equal(
            pred_unmatched.numpy(), numpy_result.pred_unmatched_mask
        )
        np.testing.assert_array_equal(
            gt_unmatched.numpy(), numpy_result.gt_unmatched_mask
        )

    def test_match_onsets_tensorflow_batch(self):
        pred_times = tf.constant(
            [[0.5, 1.0], [10.0, 11.0]],
            dtype=tf.float32,
        )
        gt_times = tf.constant(
            [[0.51, 1.01, 0.0], [10.01, 11.01, 0.0]],
            dtype=tf.float32,
        )
        gt_mask = tf.constant(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=tf.float32,
        )
        pred_confidence = tf.constant([[0.8, 0.7], [0.6, 0.5]], dtype=tf.float32)

        _, _, num_matches, pred_unmatched, _ = matching.match_onsets(
            pred_times,
            gt_times,
            gt_mask,
            pred_confidence=pred_confidence,
        )

        self.assertEqual(num_matches.numpy().tolist(), [2, 2])
        self.assertFalse(pred_unmatched.numpy().any())

    def test_assign_onset_pairs_ordered_pairs_by_sorted_gt_index(self):
        pred_times = np.array([9.0, 0.5], dtype=np.float64)
        gt_times = np.array([0.5, 3.0, 0.0], dtype=np.float64)
        gt_mask = np.array([1.0, 1.0, 0.0], dtype=np.float64)

        result = matching.assign_onset_pairs_ordered_numpy(
            pred_times, gt_times, gt_mask
        )

        self.assertEqual(int(result.num_matches[0]), 2)
        self.assertEqual(result.matched_pred_indices[0].tolist(), [0, 1])
        self.assertEqual(result.matched_gt_indices[0].tolist(), [0, 1])
        self.assertFalse(result.pred_unmatched_mask[0].any())

    def test_assign_onset_pairs_ordered_marks_extra_gt_unmatched(self):
        pred_times = np.array([0.5], dtype=np.float64)
        gt_times = np.array([0.5, 1.0], dtype=np.float64)
        gt_mask = np.array([1.0, 1.0], dtype=np.float64)

        result = matching.assign_onset_pairs_ordered_numpy(
            pred_times, gt_times, gt_mask
        )

        self.assertEqual(int(result.num_matches[0]), 1)
        self.assertTrue(result.gt_unmatched_mask[0, 1])

    def test_assign_onset_pairs_ordered_tensorflow_matches_numpy(self):
        pred_times = np.array([[0.1, 0.3, 0.5]], dtype=np.float64)
        gt_times = np.array([[0.1, 0.3, 0.5, 0.0]], dtype=np.float64)
        gt_mask = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float64)
        numpy_result = matching.assign_onset_pairs_ordered_numpy(
            pred_times, gt_times, gt_mask
        )
        tf_result = matching.assign_onset_pairs_ordered(
            tf.constant(pred_times, dtype=tf.float32),
            tf.constant(gt_times, dtype=tf.float32),
            tf.constant(gt_mask, dtype=tf.float32),
        )

        matched_pred, matched_gt, num_matches, pred_unmatched, gt_unmatched = tf_result
        self.assertEqual(int(num_matches.numpy()[0]), int(numpy_result.num_matches[0]))
        np.testing.assert_array_equal(
            matched_pred.numpy(), numpy_result.matched_pred_indices
        )
        np.testing.assert_array_equal(
            matched_gt.numpy(), numpy_result.matched_gt_indices
        )
        np.testing.assert_array_equal(
            pred_unmatched.numpy(), numpy_result.pred_unmatched_mask
        )
        np.testing.assert_array_equal(
            gt_unmatched.numpy(), numpy_result.gt_unmatched_mask
        )


if __name__ == "__main__":
    unittest.main()
