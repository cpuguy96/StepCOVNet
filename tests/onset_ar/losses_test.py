import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import losses, targets


class LossesTest(unittest.TestCase):
    def test_build_token_class_weights_upweights_rare_tokens(self) -> None:
        decoder_target_ids = np.asarray([83, 83, 83, 50], dtype=np.int32)
        decoder_mask = np.ones(4, dtype=np.float32)
        weights = losses.build_token_class_weights_numpy(
            decoder_target_ids,
            decoder_mask,
            vocab_size=100,
            scheme="inverse_freq",
        )
        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertGreater(float(weights[50]), float(weights[83]))

    def test_argmax_and_soft_predicted_times_differ_for_uniform_logits(self) -> None:
        pointer_logits = tf.zeros((1, 2, 4), dtype=tf.float32)
        residual_sec = tf.zeros((1, 2), dtype=tf.float32)
        soft = losses.predicted_times_from_outputs(
            pointer_logits,
            residual_sec,
            patch_frames=8,
            hop_sec=0.01,
            use_soft_expected=True,
        )
        hard = losses.predicted_times_from_outputs(
            pointer_logits,
            residual_sec,
            patch_frames=8,
            hop_sec=0.01,
            use_soft_expected=False,
        )
        self.assertNotAlmostEqual(
            float(soft[0, 0].numpy()),
            float(hard[0, 0].numpy()),
        )

    def test_class_weighted_token_loss_penalizes_majority_mismatch(self) -> None:
        vocab_size = targets.DeltaBucketVocab().vocab_size
        decoder_target_ids = tf.constant([[83, 50]], dtype=tf.int32)
        decoder_mask = tf.constant([[1.0, 1.0]], dtype=tf.float32)
        weights_np = losses.build_token_class_weights_numpy(
            decoder_target_ids.numpy()[0],
            decoder_mask.numpy()[0],
            vocab_size=vocab_size,
            scheme="inverse_freq",
        )
        assert weights_np is not None
        token_class_weights = tf.constant(weights_np, dtype=tf.float32)
        logits_majority = tf.constant(
            [[[0.0] * vocab_size, [0.0] * vocab_size]],
            dtype=tf.float32,
        )
        logits_majority = tf.tensor_scatter_nd_update(
            logits_majority,
            [[0, 0, 83], [0, 1, 83]],
            [10.0, 10.0],
        )
        outputs = {
            "token_logits": logits_majority,
            "pointer_logits": tf.zeros((1, 2, 8), dtype=tf.float32),
            "residual_sec": tf.zeros((1, 2), dtype=tf.float32),
        }
        batch = {
            "decoder_target_ids": decoder_target_ids,
            "decoder_mask": decoder_mask,
            "onset_step_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "target_patch_indices": tf.constant([[0, 1]], dtype=tf.int32),
            "target_times": tf.constant([[0.0, 0.1]], dtype=tf.float32),
            "target_residual_sec": tf.constant([[0.0, 0.02]], dtype=tf.float32),
        }
        unweighted, _ = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=0.0,
            lambda_residual=0.0,
            length_normalize_ce=True,
        )
        weighted, _ = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=0.0,
            lambda_residual=0.0,
            length_normalize_ce=True,
            token_class_weights=token_class_weights,
        )
        self.assertGreater(float(unweighted.numpy()), float(weighted.numpy()))

    def test_residual_loss_is_zero_when_targets_match(self) -> None:
        residual = tf.constant([[0.01, 0.02]], dtype=tf.float32)
        outputs = {
            "token_logits": tf.zeros((1, 2, 8), dtype=tf.float32),
            "pointer_logits": tf.zeros((1, 2, 8), dtype=tf.float32),
            "residual_sec": residual,
        }
        batch = {
            "decoder_target_ids": tf.constant([[1, 2]], dtype=tf.int32),
            "decoder_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "onset_step_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "target_patch_indices": tf.constant([[0, 1]], dtype=tf.int32),
            "target_times": tf.constant([[0.0, 0.1]], dtype=tf.float32),
            "target_residual_sec": residual,
        }
        _, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=0.0,
            lambda_residual=5.0,
            length_normalize_ce=True,
        )
        self.assertAlmostEqual(float(parts["residual_loss"].numpy()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
