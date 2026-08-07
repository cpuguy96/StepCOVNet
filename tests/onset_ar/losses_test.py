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

    def test_ste_matches_hard_forward_and_gives_pointer_grads(self) -> None:
        tf.random.set_seed(0)
        pointer_logits = tf.Variable(tf.random.normal((1, 2, 8), dtype=tf.float32))
        residual_sec = tf.Variable(tf.constant([[0.02, 0.03]], dtype=tf.float32))
        batch = {
            "decoder_target_ids": tf.constant([[1, 2]], dtype=tf.int32),
            "decoder_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "onset_step_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "target_patch_indices": tf.constant([[3, 5]], dtype=tf.int32),
            "target_times": tf.constant([[0.3, 0.5]], dtype=tf.float32),
            "target_residual_sec": tf.constant([[0.02, 0.03]], dtype=tf.float32),
        }
        outputs = {
            "token_logits": tf.zeros((1, 2, 10), dtype=tf.float32),
            "pointer_logits": pointer_logits,
            "residual_sec": residual_sec,
        }
        hard_times = losses.predicted_times_from_outputs(
            pointer_logits,
            residual_sec,
            patch_frames=8,
            hop_sec=0.01,
            use_soft_expected=False,
        )
        ste_times = losses.predicted_times_from_outputs(
            pointer_logits,
            residual_sec,
            patch_frames=8,
            hop_sec=0.01,
            use_ste=True,
        )
        np.testing.assert_allclose(
            ste_times.numpy(),
            hard_times.numpy(),
            rtol=0,
            atol=1e-6,
        )
        with tf.GradientTape() as tape_hard:
            total_hard, _ = losses.compute_ar_onset_loss(
                outputs,
                batch,
                patch_frames=8,
                hop_sec=0.01,
                lambda_time=1.0,
                lambda_residual=0.0,
                pointer_loss_weight=0.0,
                length_normalize_ce=True,
                use_soft_pointer_time=False,
                use_ste_pointer_time=False,
            )
        g_hard = tape_hard.gradient(total_hard, pointer_logits)
        with tf.GradientTape() as tape_ste:
            total_ste, _ = losses.compute_ar_onset_loss(
                outputs,
                batch,
                patch_frames=8,
                hop_sec=0.01,
                lambda_time=1.0,
                lambda_residual=0.0,
                pointer_loss_weight=0.0,
                length_normalize_ce=True,
                use_soft_pointer_time=False,
                use_ste_pointer_time=True,
            )
        g_ste = tape_ste.gradient(total_ste, pointer_logits)
        self.assertAlmostEqual(float(tf.norm(g_hard)), 0.0, places=6)
        self.assertGreater(float(tf.norm(g_ste)), 0.0)

    def test_local_ce_mask_keeps_target_window(self) -> None:
        logits = tf.zeros((1, 1, 16), dtype=tf.float32)
        targets_p = tf.constant([[8]], dtype=tf.int32)
        masked = losses.apply_local_pointer_ce_mask(logits, targets_p, radius=2)
        # In-window positions stay 0; outside become large negative.
        self.assertAlmostEqual(float(masked[0, 0, 8]), 0.0, places=5)
        self.assertAlmostEqual(float(masked[0, 0, 6]), 0.0, places=5)
        self.assertLess(float(masked[0, 0, 5]), -1e8)
        self.assertLess(float(masked[0, 0, 11]), -1e8)

    def test_prev_relative_window_masks_beyond_max_ahead(self) -> None:
        from stepcovnet.onset_ar import pointer_mask

        logits = tf.zeros((1, 1, 16), dtype=tf.float32)
        prev = tf.constant([[4]], dtype=tf.int32)
        masked = pointer_mask.apply_prev_relative_window_tf(
            logits,
            prev,
            max_ahead=3,
        )
        self.assertAlmostEqual(float(masked[0, 0, 4]), 0.0, places=5)
        self.assertAlmostEqual(float(masked[0, 0, 7]), 0.0, places=5)
        self.assertLess(float(masked[0, 0, 8]), -1e8)

    def test_prev_relative_window_skips_upper_bound_when_prev_zero(self) -> None:
        from stepcovnet.onset_ar import pointer_mask

        logits = tf.zeros((1, 1, 16), dtype=tf.float32)
        prev = tf.constant([[0]], dtype=tf.int32)
        masked = pointer_mask.apply_prev_relative_window_tf(
            logits,
            prev,
            max_ahead=3,
        )
        # First onset may land far past max_ahead; do not mask it.
        self.assertAlmostEqual(float(masked[0, 0, 15]), 0.0, places=5)

    def test_prev_relative_ce_step_mask_drops_gap_beyond_max_ahead(self) -> None:
        from stepcovnet.onset_ar import pointer_mask

        # Step0: first onset (prev=0) kept. Step1: gap 5 > max_ahead 3 dropped.
        targets = tf.constant([[10, 15]], dtype=tf.int32)
        prev = tf.constant([[0, 10]], dtype=tf.int32)
        onset = tf.constant([[1.0, 1.0]], dtype=tf.float32)
        mask = pointer_mask.prev_relative_ce_step_mask(
            targets,
            prev,
            max_ahead=3,
            onset_step_mask=onset,
        )
        self.assertAlmostEqual(float(mask[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(mask[0, 1]), 0.0, places=5)

    def test_soft_distance_prior_penalizes_ahead_keeps_far_finite(self) -> None:
        from stepcovnet.onset_ar import pointer_mask

        logits = tf.zeros((1, 1, 16), dtype=tf.float32)
        prev = tf.constant([[4]], dtype=tf.int32)
        soft = pointer_mask.apply_soft_distance_prior_tf(
            logits,
            prev,
            alpha=0.5,
        )
        # At prev: no penalty. +4 ahead: -2.0. Far patch stays finite (not -1e9).
        self.assertAlmostEqual(float(soft[0, 0, 4]), 0.0, places=5)
        self.assertAlmostEqual(float(soft[0, 0, 8]), -2.0, places=5)
        self.assertGreater(float(soft[0, 0, 15]), -100.0)

    def test_soft_distance_prior_skips_when_prev_zero(self) -> None:
        from stepcovnet.onset_ar import pointer_mask

        logits = tf.zeros((1, 1, 16), dtype=tf.float32)
        prev = tf.constant([[0]], dtype=tf.int32)
        soft = pointer_mask.apply_soft_distance_prior_tf(
            logits,
            prev,
            alpha=0.5,
        )
        self.assertAlmostEqual(float(soft[0, 0, 15]), 0.0, places=5)

    def test_time_loss_correct_patch_only_ignores_wrong_patches(self) -> None:
        # Argmax prefers patch 0; target is patch 3 → time term must be zero.
        pointer_logits = tf.constant(
            [[[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            dtype=tf.float32,
        )
        residual = tf.constant([[0.04]], dtype=tf.float32)
        outputs = {
            "token_logits": tf.zeros((1, 1, 8), dtype=tf.float32),
            "pointer_logits": pointer_logits,
            "residual_sec": residual,
        }
        batch = {
            "decoder_target_ids": tf.constant([[1]], dtype=tf.int32),
            "decoder_mask": tf.constant([[1.0]], dtype=tf.float32),
            "onset_step_mask": tf.constant([[1.0]], dtype=tf.float32),
            "target_patch_indices": tf.constant([[3]], dtype=tf.int32),
            "target_times": tf.constant([[0.5]], dtype=tf.float32),
            "target_residual_sec": residual,
        }
        _, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=1.0,
            lambda_residual=0.0,
            pointer_loss_weight=0.0,
            length_normalize_ce=True,
            time_loss_correct_patch_only=True,
        )
        self.assertAlmostEqual(float(parts["time_loss"].numpy()), 0.0, places=6)

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
        enc_out = encoder(
            {
                "mert_patches": tf.zeros((1, max_patches, patch_dim), dtype=tf.float32),
                "patch_mask": tf.concat(
                    [tf.ones((1, 3), dtype=tf.float32), tf.zeros((1, max_patches - 3))],
                    axis=1,
                ),
            },
            training=False,
        )
        memory, key_input = ar_models.unpack_encoder_outputs(enc_out)
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
            pointer_key_input=key_input,
        )
        self.assertEqual(times.shape, (1, max_dec))

    def test_gap_alignment_ce_and_resolved_times(self) -> None:
        gap_vocab = targets.PatchGapVocab(delta_max_dense=16, n_log_buckets=4)
        lookup = tf.constant(targets.gap_delta_lookup_table(gap_vocab), dtype=tf.int32)
        # Targets: patches [2, 5] → Δ [2, 3]; peak gap logits on those ids.
        gap_logits = tf.zeros((1, 2, gap_vocab.vocab_size), dtype=tf.float32)
        gap_logits = tf.tensor_scatter_nd_update(
            gap_logits,
            [[0, 0, 2], [0, 1, 3]],
            [20.0, 20.0],
        )
        residual = tf.constant([[0.01, 0.02]], dtype=tf.float32)
        outputs = {
            "token_logits": tf.zeros((1, 2, 8), dtype=tf.float32),
            "gap_logits": gap_logits,
            "residual_sec": residual,
        }
        batch = {
            "decoder_target_ids": tf.constant([[1, 2]], dtype=tf.int32),
            "decoder_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "onset_step_mask": tf.constant([[1.0, 1.0]], dtype=tf.float32),
            "target_patch_indices": tf.constant([[2, 5]], dtype=tf.int32),
            "target_gap_ids": tf.constant([[2, 3]], dtype=tf.int32),
            "target_times": tf.constant([[0.17, 0.42]], dtype=tf.float32),
            "target_residual_sec": residual,
            "patch_mask": tf.ones((1, 16), dtype=tf.float32),
        }
        total, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=1.0,
            lambda_residual=1.0,
            pointer_loss_weight=0.0,
            length_normalize_ce=True,
            gap_alignment=True,
            gap_loss_weight=1.0,
            gap_delta_lookup=lookup,
        )
        self.assertLess(float(parts["gap_loss"].numpy()), 0.01)
        self.assertAlmostEqual(
            float(parts["pointer_loss"].numpy()), float(parts["gap_loss"].numpy())
        )
        self.assertLess(float(parts["time_loss"].numpy()), 1e-5)
        self.assertAlmostEqual(float(parts["residual_loss"].numpy()), 0.0, places=6)
        self.assertGreater(float(total.numpy()), 0.0)

    def test_gap_alignment_ignores_soft_distance_prior(self) -> None:
        gap_vocab = targets.PatchGapVocab(delta_max_dense=8, n_log_buckets=2)
        lookup = tf.constant(targets.gap_delta_lookup_table(gap_vocab), dtype=tf.int32)
        gap_logits = tf.zeros((1, 1, gap_vocab.vocab_size), dtype=tf.float32)
        gap_logits = tf.tensor_scatter_nd_update(gap_logits, [[0, 0, 1]], [10.0])
        outputs = {
            "token_logits": tf.zeros((1, 1, 4), dtype=tf.float32),
            "gap_logits": gap_logits,
            "residual_sec": tf.zeros((1, 1), dtype=tf.float32),
        }
        batch = {
            "decoder_target_ids": tf.constant([[1]], dtype=tf.int32),
            "decoder_mask": tf.constant([[1.0]], dtype=tf.float32),
            "onset_step_mask": tf.constant([[1.0]], dtype=tf.float32),
            "target_patch_indices": tf.constant([[1]], dtype=tf.int32),
            "target_gap_ids": tf.constant([[1]], dtype=tf.int32),
            "target_times": tf.constant([[0.08]], dtype=tf.float32),
            "target_residual_sec": tf.zeros((1, 1), dtype=tf.float32),
            "patch_mask": tf.ones((1, 8), dtype=tf.float32),
        }
        _, parts = losses.compute_ar_onset_loss(
            outputs,
            batch,
            patch_frames=8,
            hop_sec=0.01,
            lambda_time=0.0,
            lambda_residual=0.0,
            pointer_loss_weight=0.0,
            length_normalize_ce=True,
            gap_alignment=True,
            gap_delta_lookup=lookup,
            pointer_soft_distance_alpha=100.0,
            pointer_local_ce_radius=1,
        )
        self.assertLess(float(parts["gap_loss"].numpy()), 0.01)


if __name__ == "__main__":
    unittest.main()
