import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config as ar_config
from stepcovnet.onset_ar import losses, targets
from stepcovnet.onset_ar import models as ar_models


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

    def test_eos_weight_scale_reduces_eos_weight(self) -> None:
        decoder_target_ids = np.asarray([83, targets.EOS_ID], dtype=np.int32)
        decoder_mask = np.ones(2, dtype=np.float32)
        base = losses.build_token_class_weights_numpy(
            decoder_target_ids,
            decoder_mask,
            vocab_size=100,
            scheme="inverse_freq",
            eos_token_weight_scale=1.0,
        )
        scaled = losses.build_token_class_weights_numpy(
            decoder_target_ids,
            decoder_mask,
            vocab_size=100,
            scheme="inverse_freq",
            eos_token_weight_scale=0.2,
        )
        assert base is not None and scaled is not None
        self.assertAlmostEqual(
            float(scaled[targets.EOS_ID]),
            float(base[targets.EOS_ID]) * 0.2,
        )

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
            pointer_loss_weight=1.0,
            length_normalize_ce=True,
        )
        weighted, _ = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=0.0,
            lambda_residual=0.0,
            pointer_loss_weight=1.0,
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
            pointer_loss_weight=1.0,
            length_normalize_ce=True,
        )
        self.assertAlmostEqual(float(parts["residual_loss"].numpy()), 0.0, places=6)

    def test_incremental_consistency_loss_zero_when_times_match(self) -> None:
        parallel = tf.constant([[0.1, 0.2, 0.0]], dtype=tf.float32)
        incremental = tf.constant([[0.1, 0.2, 0.0]], dtype=tf.float32)
        mask = tf.constant([[1.0, 1.0, 0.0]], dtype=tf.float32)
        loss = losses.incremental_consistency_loss(parallel, incremental, mask)
        self.assertAlmostEqual(float(loss.numpy()), 0.0, places=6)

    def test_compute_ar_onset_loss_accepts_float16_logits(self) -> None:
        outputs = {
            "token_logits": tf.zeros((1, 2, 8), dtype=tf.float16),
            "pointer_logits": tf.zeros((1, 2, 8), dtype=tf.float16),
            "residual_sec": tf.zeros((1, 2), dtype=tf.float16),
        }
        batch = {
            "decoder_target_ids": tf.constant([[1, 2]], dtype=tf.int32),
            "decoder_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "onset_step_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "target_patch_indices": tf.constant([[0, 1]], dtype=tf.int32),
            "target_times": tf.constant([[0.0, 0.1]], dtype=tf.float32),
            "target_residual_sec": tf.constant([[0.0, 0.02]], dtype=tf.float32),
        }
        total_loss, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=1.0,
            lambda_residual=1.0,
            pointer_loss_weight=1.0,
            length_normalize_ce=True,
        )
        self.assertEqual(total_loss.dtype, tf.float32)
        self.assertEqual(parts["token_loss"].dtype, tf.float32)

        experiment_config = ar_config.ArExperimentConfig(
            dataset=ar_config.ArDatasetConfig(max_audio_seconds=1.0, hop_sec=0.01),
            model=ar_config.ArModelConfig(
                patch_frames=4,
                d_model=32,
                n_enc_layers=1,
                n_dec_layers=2,
                num_heads=2,
                max_decode_steps=16,
                dropout_rate=0.0,
            ),
            run=ar_config.ArRunConfig(epochs=1, model_output_dir=""),
        )
        model = ar_models.build_ar_onset_model(experiment_config)
        encoder, decoder = ar_models.build_ar_onset_inference_models(
            model,
            experiment_config,
        )
        max_dec = experiment_config.max_decoder_len()
        max_patches = experiment_config.max_encoder_patches()
        patch_dim = experiment_config.patch_input_dim()
        memory = encoder(
            {
                "mert_patches": tf.zeros((1, max_patches, patch_dim), dtype=tf.float32),
                "patch_mask": tf.concat(
                    [tf.ones((1, 3), dtype=tf.float32), tf.zeros((1, max_patches - 3))],
                    axis=1,
                ),
            },
            training=False,
        )
        dec_in = tf.zeros((1, max_dec), dtype=tf.int32)
        dec_mask = tf.concat(
            [
                tf.ones((1, 4), dtype=tf.float32),
                tf.zeros((1, max_dec - 4), dtype=tf.float32),
            ],
            axis=1,
        )
        times = losses.incremental_predicted_times_tf(
            decoder,
            memory,
            tf.concat(
                [tf.ones((1, 3), dtype=tf.float32), tf.zeros((1, max_patches - 3))],
                axis=1,
            ),
            dec_in,
            dec_mask,
            max_decoder_len=max_dec,
            patch_frames=4,
            hop_sec=0.01,
            max_unroll_steps=4,
        )
        self.assertEqual(times.shape, (1, max_dec))


if __name__ == "__main__":
    unittest.main()
