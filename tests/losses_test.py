"""Tests for stepcovnet.losses (arrow focal, label-smoothed CE, masked MSE aux interval)."""

import math
import unittest

import keras
import tensorflow as tf

from stepcovnet import config, constants, losses


class SparseFocalLossTest(unittest.TestCase):
    """Tests for sparse_focal_loss."""

    def test_returns_scalar_and_masks_ignore_class(self):
        """sparse_focal_loss returns a scalar and ignores steps with y_true==ignore_class (0)."""
        batch_size, steps, num_classes = 2, 10, constants.N_ARROW_TYPES
        y_true = tf.constant(
            [[0, 1, 2, 0, 3, 0, 1, 0, 2, 1]], dtype=tf.int32
        )  # 0 = padding
        y_pred = tf.random.uniform((batch_size, steps, num_classes))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)
        self.assertLess(
            float(loss), 50.0
        )  # focal with gamma=2 stays in reasonable range

    def test_all_ignore_class_returns_finite_scalar(self):
        """When all positions are ignore_class, loss is 0 (no divide-by-zero)."""
        batch_size, steps, num_classes = 1, 4, constants.N_ARROW_TYPES
        y_true = tf.constant([[0, 0, 0, 0]], dtype=tf.int32)
        y_pred = tf.random.uniform((batch_size, steps, num_classes))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_custom_ignore_class(self):
        """ignore_class can be set to a non-default value."""
        batch_size, steps, num_classes = 1, 3, constants.N_ARROW_TYPES
        # Class 255 is ignore; only class 1 and 2 contribute
        y_true = tf.constant([[255, 1, 2]], dtype=tf.int32)
        y_pred = tf.random.uniform((batch_size, steps, num_classes))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=1.0, ignore_class=255)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_perfect_prediction_single_step_gives_near_zero_loss(self):
        """One step, p_t=1 for true class: focal loss (1-p)^gamma * (-log p) is near 0 (p clipped)."""
        n = constants.N_ARROW_TYPES
        y_true = tf.constant([[1]], dtype=tf.int32)  # true class index 1
        y_pred = tf.constant(
            [[[0.0, 1.0] + [0.0] * (n - 2)]], dtype=tf.float32
        )  # perfect at class 1
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        self.assertEqual(loss.shape, ())
        self.assertLess(float(loss), 1e-4)

    def test_single_step_exact_focal_formula(self):
        """Single valid step: loss = (1 - p_t)^gamma * (-log(p_t)); p_t=0.25, gamma=2."""
        n = constants.N_ARROW_TYPES
        p_t = 0.25
        remainder = (1.0 - p_t) / (n - 1)
        row = [remainder] * n
        row[1] = p_t
        y_true = tf.constant([[1]], dtype=tf.int32)
        y_pred = tf.constant([[[row]]], shape=(1, 1, n), dtype=tf.float32)
        loss = losses.sparse_focal_loss(y_true, y_pred, gamma=2.0, ignore_class=0)
        expected = (1.0 - p_t) ** 2 * (-math.log(p_t))
        self.assertAlmostEqual(float(loss), expected, places=4)
        self.assertGreater(float(loss), 0.5)
        self.assertLess(float(loss), 1.0)


class ArrowLabelSmoothedCrossentropyTest(unittest.TestCase):
    """Tests for arrow_label_smoothed_crossentropy."""

    def test_returns_scalar(self):
        """arrow_label_smoothed_crossentropy returns a scalar loss."""
        batch_size, steps, num_classes = 2, 5, constants.N_ARROW_TYPES
        y_true = tf.constant([[1, 2, 3, 1, 2], [2, 1, 3, 2, 1]], dtype=tf.int32)
        y_pred = tf.random.uniform((batch_size, steps, num_classes))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.1)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)
        self.assertLess(float(loss), 20.0)

    def test_masks_padding(self):
        """arrow_label_smoothed_crossentropy ignores steps where y_true is ARROW_PADDING_CLASS."""
        y_true = tf.constant([[constants.ARROW_PADDING_CLASS, 1]], dtype=tf.int32)
        num_classes = constants.N_ARROW_TYPES
        uniform = [1.0 / num_classes] * num_classes
        perfect_step1 = [0.0, 1.0] + [0.0] * (num_classes - 2)
        y_pred = tf.constant(
            [[uniform, perfect_step1]],
            dtype=tf.float32,
        )
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.0)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_single_step_smoothing_zero_exact_ce(self):
        """One step, smoothing=0: loss = -log(p_true). For p_true=0.5 at true class 1 expect -log(0.5)=ln(2)."""
        n = constants.N_ARROW_TYPES
        p_true = 0.5
        true_class = 1  # 0 is padding and would be masked
        remainder = (1.0 - p_true) / (n - 1)
        row = [remainder] * n
        row[true_class] = p_true
        y_true = tf.constant([[true_class]], dtype=tf.int32)
        y_pred = tf.constant([[[row]]], shape=(1, 1, n), dtype=tf.float32)
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.0)
        expected = -math.log(p_true)
        self.assertAlmostEqual(float(loss), expected, places=4)
        self.assertAlmostEqual(float(loss), math.log(2), places=4)

    def test_all_padding_returns_zero(self):
        """With all padding, loss is 0 (no divide-by-zero)."""
        y_true = tf.constant(
            [[constants.ARROW_PADDING_CLASS, constants.ARROW_PADDING_CLASS]],
            dtype=tf.int32,
        )
        y_pred = tf.random.uniform((1, 2, constants.N_ARROW_TYPES))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = losses.arrow_label_smoothed_crossentropy(y_true, y_pred, smoothing=0.1)
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_smoothing_zero_vs_positive(self):
        """smoothing=0 and 0.1 give different losses."""
        y_true = tf.constant([[1, 2, 3]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 3, constants.N_ARROW_TYPES))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss_zero = losses.arrow_label_smoothed_crossentropy(
            y_true, y_pred, smoothing=0.0
        )
        loss_smooth = losses.arrow_label_smoothed_crossentropy(
            y_true, y_pred, smoothing=0.1
        )
        self.assertNotEqual(float(loss_zero), float(loss_smooth))
        self.assertGreater(float(loss_zero), 0.0)
        self.assertGreater(float(loss_smooth), 0.0)

    def test_perfect_prediction_lower_than_random(self):
        """Loss is lower for correct predictions than uniform."""
        y_true = tf.constant([[1, 2]], dtype=tf.int32)
        num_classes = constants.N_ARROW_TYPES
        perfect_pred = tf.constant(
            [
                [
                    [0.0, 1.0] + [0.0] * (num_classes - 2),
                    [0.0, 0.0, 1.0] + [0.0] * (num_classes - 3),
                ]
            ],
            dtype=tf.float32,
        )
        uniform_pred = tf.ones((1, 2, num_classes), dtype=tf.float32) / num_classes
        loss_perfect = losses.arrow_label_smoothed_crossentropy(
            y_true, perfect_pred, smoothing=0.1
        )
        loss_uniform = losses.arrow_label_smoothed_crossentropy(
            y_true, uniform_pred, smoothing=0.1
        )
        self.assertLess(float(loss_perfect), float(loss_uniform))
        self.assertGreater(float(loss_uniform), float(loss_perfect) + 0.1)


class MaskedMseAuxIntervalTest(unittest.TestCase):
    """Tests for masked_mse_aux_interval."""

    def test_without_sample_weight_returns_mean_over_all_elements(self):
        """Without sample_weight returns mean squared error over all elements."""
        y_true = tf.constant([[[0.1], [0.2], [0.3]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.2], [0.2], [0.4]]], dtype=tf.float32)
        loss = losses.masked_mse_aux_interval(y_true, y_pred, sample_weight=None)
        self.assertEqual(loss.shape, ())
        # Errors: 0.1, 0.0, 0.1 -> squared: 0.01, 0, 0.01 -> mean = 0.02/3
        expected = ((0.2 - 0.1) ** 2 + (0.2 - 0.2) ** 2 + (0.4 - 0.3) ** 2) / 3
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertAlmostEqual(float(loss), 0.02 / 3, places=6)

    def test_with_sample_weight_masks_steps(self):
        """With sample_weight only averages over masked (1.0) steps: MSE = (1-1)^2 + (3-2)^2 over 2 steps."""
        y_true = tf.constant([[[1.0], [2.0], [0.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[1.0], [3.0], [99.0]]], dtype=tf.float32)
        sample_weight = tf.constant([[[1.0], [1.0], [0.0]]], dtype=tf.float32)
        loss = losses.masked_mse_aux_interval(
            y_true, y_pred, sample_weight=sample_weight
        )
        self.assertEqual(loss.shape, ())
        # Only steps 0 and 1 count: (1-1)^2 + (3-2)^2 = 0 + 1, mean = 0.5
        self.assertAlmostEqual(float(loss), 0.5, places=5)
        self.assertGreater(float(loss), 0.0)

    def test_single_element_exact_mse(self):
        """Single element: loss = (y_pred - y_true)^2."""
        y_true = tf.constant([[[3.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[5.0]]], dtype=tf.float32)
        loss = losses.masked_mse_aux_interval(y_true, y_pred, sample_weight=None)
        self.assertAlmostEqual(float(loss), (5.0 - 3.0) ** 2, places=5)
        self.assertAlmostEqual(float(loss), 4.0, places=5)

    def test_sample_weight_all_zero_returns_finite_scalar(self):
        """When sample_weight sum is zero, loss is 0 (no divide-by-zero)."""
        y_true = tf.constant([[[1.0], [2.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[2.0], [3.0]]], dtype=tf.float32)
        sample_weight = tf.constant([[[0.0], [0.0]]], dtype=tf.float32)
        loss = losses.masked_mse_aux_interval(
            y_true, y_pred, sample_weight=sample_weight
        )
        self.assertEqual(loss.shape, ())
        self.assertAlmostEqual(float(loss), 0.0, places=5)


class BuildArrowCombinedLossTest(unittest.TestCase):
    """Tests for build_arrow_combined_loss."""

    def _run_config(
        self,
        *,
        loss_type: str = "crossentropy",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        chart_validity_aux_weight: float = 0.0,
        diversity_aux_weight: float = 0.0,
        chart_validity_rejection_threshold: float | None = None,
        chart_validity_rejection_scale: float = 10.0,
        chart_validity_rejection_temperature: float = 50.0,
    ) -> config.ArrowRunConfig:
        return config.ArrowRunConfig(
            epoch=1,
            take_count=1,
            model_output_dir="/tmp",
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            chart_validity_aux_weight=chart_validity_aux_weight,
            diversity_aux_weight=diversity_aux_weight,
            chart_validity_rejection_threshold=chart_validity_rejection_threshold,
            chart_validity_rejection_scale=chart_validity_rejection_scale,
            chart_validity_rejection_temperature=chart_validity_rejection_temperature,
        )

    def test_returns_callable_that_produces_scalar_crossentropy(self):
        """build_arrow_combined_loss returns a callable; crossentropy path yields scalar loss."""
        run_config = self._run_config(loss_type="crossentropy")
        combined = losses.build_arrow_combined_loss(run_config)
        y_true = tf.constant([[1, 2, 1]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 3, constants.N_ARROW_TYPES))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = combined(y_true, y_pred)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_focal_path_produces_scalar(self):
        """Combined loss with loss_type=focal uses focal main loss and returns scalar."""
        run_config = self._run_config(loss_type="focal", focal_gamma=2.0)
        combined = losses.build_arrow_combined_loss(run_config)
        y_true = tf.constant([[1, 2]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 2, constants.N_ARROW_TYPES))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = combined(y_true, y_pred)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_label_smoothing_path_produces_scalar(self):
        """Combined loss with label_smoothing > 0 uses smoothed crossentropy and returns scalar."""
        run_config = self._run_config(loss_type="crossentropy", label_smoothing=0.1)
        combined = losses.build_arrow_combined_loss(run_config)
        y_true = tf.constant([[1, 2]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 2, constants.N_ARROW_TYPES))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = combined(y_true, y_pred)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_rejection_threshold_path_produces_scalar(self):
        """Combined loss with chart_validity_rejection_threshold set uses tiered loss."""
        run_config = self._run_config(
            chart_validity_rejection_threshold=0.99,
            chart_validity_rejection_scale=10.0,
            chart_validity_rejection_temperature=50.0,
        )
        combined = losses.build_arrow_combined_loss(run_config)
        y_true = tf.constant([[1, 2, 1]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 3, constants.N_ARROW_TYPES))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = combined(y_true, y_pred)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_with_aux_weights_includes_validity_and_diversity(self):
        """Combined loss with positive aux weights still returns a scalar."""
        run_config = self._run_config(
            chart_validity_aux_weight=0.5,
            diversity_aux_weight=0.5,
        )
        combined = losses.build_arrow_combined_loss(run_config)
        y_true = tf.constant([[1, 2]], dtype=tf.int32)
        y_pred = tf.random.uniform((1, 2, constants.N_ARROW_TYPES))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        loss = combined(y_true, y_pred)
        self.assertEqual(loss.shape, ())
        self.assertGreater(float(loss), 0.0)

    def test_combined_equals_main_when_no_aux_and_no_rejection(self):
        """When aux weights and rejection are 0, combined loss equals main (CE or focal) loss."""
        run_config = self._run_config(
            loss_type="crossentropy",
            chart_validity_aux_weight=0.0,
            diversity_aux_weight=0.0,
            chart_validity_rejection_threshold=None,
        )
        combined = losses.build_arrow_combined_loss(run_config)
        y_true = tf.constant([[1, 2]], dtype=tf.int32)
        n = constants.N_ARROW_TYPES
        y_pred = tf.random.uniform((1, 2, n))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        combined_loss_val = float(combined(y_true, y_pred))
        ce = keras.losses.SparseCategoricalCrossentropy(
            ignore_class=constants.ARROW_PADDING_CLASS
        )
        main_loss_val = float(ce(y_true, y_pred))
        self.assertAlmostEqual(combined_loss_val, main_loss_val, places=5)

    def test_combined_focal_equals_main_when_no_aux_and_no_rejection(self):
        """When loss_type=focal and no aux/rejection, combined equals sparse_focal_loss."""
        run_config = self._run_config(
            loss_type="focal",
            focal_gamma=2.0,
            chart_validity_aux_weight=0.0,
            diversity_aux_weight=0.0,
            chart_validity_rejection_threshold=None,
        )
        combined = losses.build_arrow_combined_loss(run_config)
        y_true = tf.constant([[1, 2]], dtype=tf.int32)
        n = constants.N_ARROW_TYPES
        y_pred = tf.random.uniform((1, 2, n))
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        combined_loss_val = float(combined(y_true, y_pred))
        focal_val = float(
            losses.sparse_focal_loss(
                y_true, y_pred, gamma=2.0, ignore_class=constants.ARROW_PADDING_CLASS
            )
        )
        self.assertAlmostEqual(combined_loss_val, focal_val, places=5)
