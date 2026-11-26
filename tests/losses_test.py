import math
import unittest

import torch

from stepcovnet import config, constants, losses


def _rand_probs(*shape: int) -> torch.Tensor:
    t = torch.rand(*shape, dtype=torch.float32) + 1e-4
    return t / t.sum(dim=-1, keepdim=True)


class SparseFocalLossTest(unittest.TestCase):
    """Tests for sparse_focal_loss."""

    def test_returns_scalar_and_masks_ignore_class(self):
        batch_size, steps, num_classes = 2, 10, constants.N_ARROW_TYPES
        y_true = torch.tensor(
            [[0, 1, 2, 0, 3, 0, 1, 0, 2, 1]] * batch_size, dtype=torch.long
        )
        y_pred = _rand_probs(batch_size, steps, num_classes)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)
        self.assertLess(float(loss), 50.0)

    def test_all_ignore_class_returns_finite_scalar(self):
        y_true = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        y_pred = _rand_probs(1, 4, constants.N_ARROW_TYPES)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_custom_ignore_class(self):
        y_true = torch.tensor([[255, 1, 2]], dtype=torch.long)
        y_pred = _rand_probs(1, 3, constants.N_ARROW_TYPES)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=1.0, ignore_class=255)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_perfect_prediction_single_step_gives_near_zero_loss(self):
        n = constants.N_ARROW_TYPES
        y_true = torch.tensor([[1]], dtype=torch.long)
        y_pred = torch.tensor([[[0.0, 1.0] + [0.0] * (n - 2)]], dtype=torch.float32)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        self.assertEqual(loss.shape, ())
        self.assertLess(float(loss), 1e-4)

    def test_single_step_exact_focal_formula(self):
        n = constants.N_ARROW_TYPES
        p_t = 0.25
        remainder = (1.0 - p_t) / (n - 1)
        row = [remainder] * n
        row[1] = p_t
        y_true = torch.tensor([[1]], dtype=torch.long)
        y_pred = torch.tensor([[row]], dtype=torch.float32)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        expected = (1.0 - p_t) ** 2 * (-math.log(p_t))
        self.assertAlmostEqual(float(loss), expected, places=4)


class ArrowLabelSmoothedCrossentropyTest(unittest.TestCase):
    """Tests for arrow_label_smoothed_crossentropy."""

    def test_returns_scalar(self):
        y_true = torch.tensor([[1, 2, 3, 1, 2], [2, 1, 3, 2, 1]], dtype=torch.long)
        y_pred = _rand_probs(2, 5, constants.N_ARROW_TYPES)
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.1)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)
        self.assertLess(float(loss), 20.0)

    def test_masks_padding(self):
        y_true = torch.tensor([[constants.ARROW_PADDING_CLASS, 1]], dtype=torch.long)
        num_classes = constants.N_ARROW_TYPES
        uniform = [1.0 / num_classes] * num_classes
        perfect_step1 = [0.0, 1.0] + [0.0] * (num_classes - 2)
        y_pred = torch.tensor([[uniform, perfect_step1]], dtype=torch.float32)
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.0)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_single_step_smoothing_zero_exact_ce(self):
        n = constants.N_ARROW_TYPES
        p_true = 0.5
        true_class = 1
        remainder = (1.0 - p_true) / (n - 1)
        row = [remainder] * n
        row[true_class] = p_true
        y_true = torch.tensor([[true_class]], dtype=torch.long)
        y_pred = torch.tensor([[row]], dtype=torch.float32)
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.0)
        self.assertAlmostEqual(float(loss), -math.log(p_true), places=4)

    def test_all_padding_returns_zero(self):
        y_true = torch.tensor(
            [[constants.ARROW_PADDING_CLASS, constants.ARROW_PADDING_CLASS]],
            dtype=torch.long,
        )
        y_pred = _rand_probs(1, 2, constants.N_ARROW_TYPES)
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.1)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_smoothing_zero_vs_positive(self):
        y_true = torch.tensor([[1, 2, 3]], dtype=torch.long)
        y_pred = _rand_probs(1, 3, constants.N_ARROW_TYPES)
        loss_zero = losses.arrow_label_smoothed_crossentropy(
            y_true, y_pred, smoothing=0.0
        )
        loss_smooth = losses.arrow_label_smoothed_crossentropy(
            y_true, y_pred, smoothing=0.1
        )
        self.assertNotEqual(float(loss_zero), float(loss_smooth))

    def test_perfect_prediction_lower_than_random(self):
        y_true = torch.tensor([[1, 2]], dtype=torch.long)
        num_classes = constants.N_ARROW_TYPES
        perfect_pred = torch.tensor(
            [
                [
                    [0.0, 1.0] + [0.0] * (num_classes - 2),
                    [0.0, 0.0, 1.0] + [0.0] * (num_classes - 3),
                ]
            ],
            dtype=torch.float32,
        )
        uniform_pred = (
            torch.ones((1, 2, num_classes), dtype=torch.float32) / num_classes
        )
        loss_perfect = losses.arrow_label_smoothed_crossentropy(
            y_true, perfect_pred, smoothing=0.1
        )
        loss_uniform = losses.arrow_label_smoothed_crossentropy(
            y_true, uniform_pred, smoothing=0.1
        )
        self.assertLess(float(loss_perfect), float(loss_uniform))


class MaskedMseAuxIntervalTest(unittest.TestCase):
    """Tests for masked_mse_aux_interval."""

    def test_without_sample_weight_returns_mean_over_all_elements(self):
        y_true = torch.tensor([[[0.1], [0.2], [0.3]]], dtype=torch.float32)
        y_pred = torch.tensor([[[0.2], [0.2], [0.4]]], dtype=torch.float32)
        loss = losses.masked_mse_aux_interval(y_true, y_pred, sample_weight=None)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.02 / 3, places=6)

    def test_with_sample_weight_masks_steps(self):
        y_true = torch.tensor([[[1.0], [2.0], [0.0]]], dtype=torch.float32)
        y_pred = torch.tensor([[[1.0], [3.0], [99.0]]], dtype=torch.float32)
        sample_weight = torch.tensor([[[1.0], [1.0], [0.0]]], dtype=torch.float32)
        loss = losses.masked_mse_aux_interval(
            y_true, y_pred, sample_weight=sample_weight
        )
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.5, places=5)

    def test_sample_weight_all_zero_returns_finite_scalar(self):
        y_true = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)
        y_pred = torch.tensor([[[2.0], [3.0]]], dtype=torch.float32)
        sample_weight = torch.tensor([[[0.0], [0.0]]], dtype=torch.float32)
        loss = losses.masked_mse_aux_interval(
            y_true, y_pred, sample_weight=sample_weight
        )
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)


class BuildArrowCombinedLossTest(unittest.TestCase):
    """Tests for build_arrow_combined_loss."""

    def _run_config(self, **kwargs) -> config.ArrowRunConfig:
        return config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="/tmp",
            **kwargs,
        )

    def test_crossentropy_focal_smoothing_rejection_and_aux_paths(self):
        y_true = torch.tensor([[1, 2, 1]], dtype=torch.long)
        y_pred = _rand_probs(1, 3, constants.N_ARROW_TYPES)
        configs = [
            self._run_config(loss_type="crossentropy"),
            self._run_config(loss_type="focal", focal_gamma=2.0),
            self._run_config(loss_type="crossentropy", label_smoothing=0.1),
            self._run_config(
                chart_validity_aux_weight=0.5,
                diversity_aux_weight=0.5,
            ),
            self._run_config(
                chart_validity_rejection_threshold=0.99,
                chart_validity_rejection_scale=10.0,
                chart_validity_rejection_temperature=50.0,
            ),
        ]
        for cfg in configs:
            loss_val = losses.build_arrow_combined_loss(cfg)(y_true, y_pred)
            self.assertEqual(loss_val.shape, ())
            self.assertGreater(float(loss_val), 0.0)

    def test_combined_equals_main_when_no_aux_and_no_rejection(self):
        y_true = torch.tensor([[1, 2, 1]], dtype=torch.long)
        y_pred = _rand_probs(1, 3, constants.N_ARROW_TYPES)
        cfg = self._run_config(
            loss_type="focal",
            focal_gamma=2.0,
            chart_validity_aux_weight=0.0,
            diversity_aux_weight=0.0,
        )
        combined_val = losses.build_arrow_combined_loss(cfg)(y_true, y_pred)
        direct_val = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0)
        self.assertAlmostEqual(float(combined_val), float(direct_val), places=5)


if __name__ == "__main__":
    unittest.main()
