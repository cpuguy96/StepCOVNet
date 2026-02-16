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
        self.assertEqual(config['tolerance'], 1)
        self.assertEqual(config['threshold'], 0.5)
        self.assertEqual(config['name'], 'onset_f1_score')


if __name__ == '__main__':
    unittest.main()
