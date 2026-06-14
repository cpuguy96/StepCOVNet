import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_events import losses, matching


def _run_loss(
    pred_times,
    pred_confidence,
    gt_times,
    gt_mask,
    duration=10.0,
    **kwargs,
):
    return losses.compute_onset_event_loss(
        tf.constant(pred_times, dtype=tf.float32),
        tf.constant(pred_confidence, dtype=tf.float32),
        tf.constant(gt_times, dtype=tf.float32),
        tf.constant(gt_mask, dtype=tf.float32),
        tf.constant(duration, dtype=tf.float32),
        **kwargs,
    )


class OnsetEventLossTest(unittest.TestCase):
    def test_returns_scalar_tensor(self):
        loss = _run_loss(
            pred_times=[[1.0, 5.0]],
            pred_confidence=[[0.9, 0.1]],
            gt_times=[[1.0, 0.0]],
            gt_mask=[[1.0, 0.0]],
        )

        self.assertEqual(loss.shape, ())
        self.assertTrue(np.isfinite(float(loss.numpy())))

    def test_perfect_match_zero_time_loss(self):
        _, components = _run_loss(
            pred_times=[[1.0, 2.0]],
            pred_confidence=[[0.99, 0.01]],
            gt_times=[[1.0, 2.0, 0.0]],
            gt_mask=[[1.0, 1.0, 0.0]],
            return_components=True,
        )

        self.assertAlmostEqual(float(components["time_loss"].numpy()), 0.0, places=6)

    def test_time_loss_on_nearest_query(self):
        _, components = _run_loss(
            pred_times=[[1.01, 5.0]],
            pred_confidence=[[0.9, 0.1]],
            gt_times=[[1.0, 0.0]],
            gt_mask=[[1.0, 0.0]],
            lambda_cls=0.0,
            return_components=True,
        )

        self.assertAlmostEqual(float(components["time_loss"].numpy()), 0.0101, places=5)

    def test_time_loss_is_nearest_l1(self):
        _, components = _run_loss(
            pred_times=[[1.0, 5.0]],
            pred_confidence=[[0.9, 0.1]],
            gt_times=[[1.05, 0.0]],
            gt_mask=[[1.0, 0.0]],
            lambda_cls=0.0,
            tolerance_sec=0.02,
            return_components=True,
        )

        self.assertAlmostEqual(float(components["time_loss"].numpy()), 0.2025, places=5)
        self.assertAlmostEqual(
            float(components["total_loss"].numpy()), 1.0125, places=5
        )

    def test_no_gt_onsets_time_loss_zero(self):
        _, components = _run_loss(
            pred_times=[[1.0, 2.0]],
            pred_confidence=[[0.8, 0.7]],
            gt_times=[[0.0, 0.0]],
            gt_mask=[[0.0, 0.0]],
            return_components=True,
        )

        self.assertAlmostEqual(float(components["time_loss"].numpy()), 0.0, places=6)
        self.assertGreater(float(components["unmatched_conf_loss"].numpy()), 0.0)

    def test_far_predictions_still_drive_time_loss(self):
        # L1-Hungarian uniquely assigns: GT=2.0 → pred=0.5 (diff 1.5),
        # GT=3.0 → pred=0.0 (diff 3.0), mean = (1.5+3.0)/2 = 2.25.
        _, components = _run_loss(
            pred_times=[[0.0, 0.5]],
            pred_confidence=[[0.9, 0.8]],
            gt_times=[[2.0, 3.0]],
            gt_mask=[[1.0, 1.0]],
            tolerance_sec=0.02,
            return_components=True,
        )

        self.assertAlmostEqual(float(components["time_loss"].numpy()), 18.525, places=5)

    def test_near_predictions_small_time_loss(self):
        _, components = _run_loss(
            pred_times=[[2.0, 3.0]],
            pred_confidence=[[0.9, 0.8]],
            gt_times=[[2.01, 3.01]],
            gt_mask=[[1.0, 1.0]],
            tolerance_sec=0.02,
            return_components=True,
        )

        self.assertAlmostEqual(float(components["time_loss"].numpy()), 0.0101, places=5)

    def test_lambda_weights_scale_total_loss(self):
        _, base = _run_loss(
            pred_times=[[1.0]],
            pred_confidence=[[0.9]],
            gt_times=[[1.01]],
            gt_mask=[[1.0]],
            lambda_cls=1.0,
            lambda_time=5.0,
            return_components=True,
        )
        _, scaled = _run_loss(
            pred_times=[[1.0]],
            pred_confidence=[[0.9]],
            gt_times=[[1.01]],
            gt_mask=[[1.0]],
            lambda_cls=2.0,
            lambda_time=10.0,
            return_components=True,
        )

        matched = float(base["matched_conf_loss"].numpy())
        unmatched = float(base["unmatched_conf_loss"].numpy())
        time = float(base["time_loss"].numpy())
        expected_scaled = 2.0 * (matched + 0.25 * unmatched) + 10.0 * time
        self.assertAlmostEqual(
            float(scaled["total_loss"].numpy()), expected_scaled, places=5
        )

    def test_return_components_dict(self):
        total, components = _run_loss(
            pred_times=[[0.5, 1.0]],
            pred_confidence=[[0.95, 0.05]],
            gt_times=[[0.51, 1.01, 0.0]],
            gt_mask=[[1.0, 1.0, 0.0]],
            return_components=True,
        )

        self.assertIn("matched_conf_loss", components)
        self.assertIn("unmatched_conf_loss", components)
        self.assertIn("time_loss", components)
        self.assertIn("total_loss", components)
        self.assertAlmostEqual(
            float(total.numpy()),
            float(components["total_loss"].numpy()),
            places=6,
        )

    def test_batch_dimension(self):
        loss = _run_loss(
            pred_times=[[1.0, 5.0], [10.0, 11.0]],
            pred_confidence=[[0.9, 0.1], [0.85, 0.15]],
            gt_times=[[1.0, 0.0], [10.01, 11.01]],
            gt_mask=[[1.0, 0.0], [1.0, 1.0]],
            duration=[10.0, 20.0],
        )

        self.assertEqual(loss.shape, ())
        self.assertTrue(np.isfinite(float(loss.numpy())))

    def test_batch_time_loss_averages_over_gt(self):
        _, components = _run_loss(
            pred_times=[[1.0, 5.0], [10.0, 11.0]],
            pred_confidence=[[0.9, 0.1], [0.9, 0.9]],
            gt_times=[[1.05, 0.0], [10.05, 11.05]],
            gt_mask=[[1.0, 0.0], [1.0, 1.0]],
            lambda_cls=0.0,
            tolerance_sec=0.02,
            return_components=True,
        )

        self.assertAlmostEqual(float(components["time_loss"].numpy()), 0.2025, places=5)

    def test_hungarian_differs_from_ordered_when_times_cross(self):
        _, components = _run_loss(
            pred_times=[[5.0, 1.0]],
            pred_confidence=[[0.9, 0.9]],
            gt_times=[[1.0, 5.0, 0.0]],
            gt_mask=[[1.0, 1.0, 0.0]],
            lambda_cls=0.0,
            return_components=True,
        )
        ordered = matching.assign_onset_pairs_ordered_numpy(
            np.array([[5.0, 1.0]]),
            np.array([[1.0, 5.0, 0.0]]),
            np.array([[1.0, 1.0, 0.0]]),
        )
        hungarian = matching.assign_onset_pairs_l1_numpy(
            np.array([[5.0, 1.0]]),
            np.array([[1.0, 5.0, 0.0]]),
            np.array([[1.0, 1.0, 0.0]]),
        )

        self.assertNotEqual(
            ordered.matched_gt_indices.tolist(),
            hungarian.matched_gt_indices.tolist(),
        )
        self.assertLess(float(components["time_loss"].numpy()), 0.01)

    def test_default_tolerance_parameter(self):
        pred_times = [[1.0, 0.02]]
        gt_times = [[0.02, 0.05]]
        gt_mask = [[1.0, 1.0]]
        pred_confidence = [[0.9, 0.9]]

        _, at_default = _run_loss(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            return_components=True,
        )
        _, explicit = _run_loss(
            pred_times,
            pred_confidence,
            gt_times,
            gt_mask,
            tolerance_sec=matching.DEFAULT_TOLERANCE_SEC,
            return_components=True,
        )

        self.assertAlmostEqual(
            float(at_default["total_loss"].numpy()),
            float(explicit["total_loss"].numpy()),
            places=6,
        )

    def test_confidence_targets_follow_hungarian_matching(self):
        _, good = _run_loss(
            pred_times=[[0.5, 9.0]],
            pred_confidence=[[0.99, 0.01]],
            gt_times=[[0.5, 0.0]],
            gt_mask=[[1.0, 0.0]],
            lambda_time=0.0,
            return_components=True,
        )
        _, bad = _run_loss(
            pred_times=[[0.5, 9.0]],
            pred_confidence=[[0.01, 0.99]],
            gt_times=[[0.5, 0.0]],
            gt_mask=[[1.0, 0.0]],
            lambda_time=0.0,
            return_components=True,
        )

        self.assertLess(
            float(good["matched_conf_loss"].numpy()),
            float(bad["matched_conf_loss"].numpy()),
        )

    def test_duration_scalar_accepted(self):
        loss = _run_loss(
            pred_times=[[1.0]],
            pred_confidence=[[0.9]],
            gt_times=[[1.0]],
            gt_mask=[[1.0]],
            duration=30.0,
        )

        self.assertTrue(np.isfinite(float(loss.numpy())))


if __name__ == "__main__":
    unittest.main()
