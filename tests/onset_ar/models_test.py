import unittest

import numpy as np
import tensorflow as tf

from stepcovnet import onset_metric_names as mn
from stepcovnet import timing_match
from stepcovnet.onset_ar import config, datasets, losses, models, targets, trainers


def _tiny_experiment_config() -> config.ArExperimentConfig:
    return config.ArExperimentConfig(
        dataset=config.ArDatasetConfig(max_audio_seconds=1.0, hop_sec=0.01),
        model=config.ArModelConfig(
            patch_frames=4,
            d_model=32,
            n_enc_layers=1,
            n_dec_layers=1,
            num_heads=2,
            max_decode_steps=16,
            delta_max_dense=8,
            n_log_buckets=4,
            n_first_abs_bins=8,
        ),
        run=config.ArRunConfig(epochs=1, model_output_dir=""),
    )


class ModelsTest(unittest.TestCase):
    def test_pairwise_valid_mask_masks_padding(self) -> None:
        layer = models.PairwiseValidMask(keep_valid=True)
        valid = tf.constant([[1.0, 1.0, 0.0]], dtype=tf.float32)
        mask = layer(valid)
        self.assertTrue(bool(mask[0, 0, 0].numpy()) is True)
        self.assertTrue(bool(mask[0, 2, 2].numpy()) is False)
        self.assertTrue(bool(mask[0, 0, 2].numpy()) is False)

    def test_decoder_self_attention_masks_future(self) -> None:
        layer = models.DecoderSelfAttentionMask(max_decoder_len=4, keep_valid=True)
        decoder_mask = tf.ones((1, 4), dtype=tf.float32)
        mask = layer(decoder_mask)
        self.assertTrue(bool(mask[0, 0, 0].numpy()) is True)
        self.assertTrue(bool(mask[0, 0, 1].numpy()) is False)
        self.assertTrue(bool(mask[0, 1, 0].numpy()) is True)

    def test_build_ar_onset_model_output_shapes(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        max_patches = experiment_config.max_encoder_patches()
        max_dec = experiment_config.max_decoder_len()
        patch_dim = experiment_config.patch_input_dim()
        vocab_size = experiment_config.build_vocab().vocab_size

        inputs = {
            "mert_patches": tf.zeros((1, max_patches, patch_dim), dtype=tf.float32),
            "patch_mask": tf.concat(
                [
                    tf.ones((1, 4), dtype=tf.float32),
                    tf.zeros((1, max_patches - 4), tf.float32),
                ],
                axis=1,
            ),
            "decoder_input_ids": tf.zeros((1, max_dec), dtype=tf.int32),
            "decoder_mask": tf.concat(
                [
                    tf.ones((1, 3), dtype=tf.float32),
                    tf.zeros((1, max_dec - 3), tf.float32),
                ],
                axis=1,
            ),
        }
        outputs = model(inputs, training=False)
        self.assertEqual(outputs["token_logits"].shape, (1, max_dec, vocab_size))
        self.assertEqual(outputs["pointer_logits"].shape, (1, max_dec, max_patches))
        self.assertEqual(outputs["residual_sec"].shape, (1, max_dec))

    def test_ar_onset_model_accepts_compact_dynamic_shapes(self) -> None:
        experiment_config = _tiny_experiment_config()
        times = np.asarray([0.05, 0.12], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(3, experiment_config.patch_input_dim()).astype(
                np.float32,
            ),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )
        arrays = datasets.sample_to_training_arrays(
            sample,
            experiment_config,
            pad_to_configured_max=False,
        )
        inputs = {
            key: tf.constant(arrays[key][np.newaxis, ...])
            for key in (
                "mert_patches",
                "patch_mask",
                "decoder_input_ids",
                "decoder_mask",
            )
        }

        outputs = models.build_ar_onset_model(experiment_config)(inputs, training=False)

        self.assertEqual(outputs["token_logits"].shape[1], token_seq.n_steps + 1)
        self.assertEqual(outputs["pointer_logits"].shape, (1, token_seq.n_steps + 1, 3))
        self.assertEqual(outputs["residual_sec"].shape, (1, token_seq.n_steps + 1))


class TrainersTest(unittest.TestCase):
    def test_scheduled_sampling_applies_after_train_step_is_traced(self) -> None:
        """Regression: the ramp must survive tracing.

        ``train_step`` is traced during warmup while ``p`` is still 0. When the
        branch was a Python ``if`` the sampling path was compiled out and the
        ramp callback silently did nothing for the rest of the run.
        """
        tf.random.set_seed(0)
        np.random.seed(0)
        experiment_config = _tiny_experiment_config()
        experiment_config.run.scheduled_sampling_max_p = 1.0
        experiment_config.run.scheduled_sampling_warmup_epochs = 1
        experiment_config.run.scheduled_sampling_ramp_epochs = 1
        times = np.asarray([0.05, 0.12], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(3, experiment_config.patch_input_dim()).astype(
                np.float32,
            ),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )
        batch = datasets.sample_to_training_batch(sample, experiment_config)
        tf_batch = {key: tf.constant(value) for key, value in batch.items()}

        training_model = trainers.ArOnsetTrainingModel(
            models.build_ar_onset_model(experiment_config),
            experiment_config=experiment_config,
        )

        # Closure, not a bound method: this is the form Keras uses to build the
        # train function, and the only one that actually traces a graph here.
        @tf.function
        def select_inputs(batch: dict[str, tf.Tensor]) -> tf.Tensor:
            return training_model._decoder_inputs_for_step(batch)

        teacher_inputs = tf_batch["decoder_input_ids"].numpy()
        during_warmup = select_inputs(tf_batch).numpy()
        np.testing.assert_array_equal(during_warmup, teacher_inputs)

        training_model.scheduled_sampling_p.assign(1.0)
        after_ramp = select_inputs(tf_batch).numpy()
        self.assertFalse(
            np.array_equal(after_ramp, teacher_inputs),
            "scheduled sampling did not change decoder inputs after the ramp",
        )

    def test_scheduled_sampling_ramp_callback_updates_variable(self) -> None:
        experiment_config = _tiny_experiment_config()
        experiment_config.run.scheduled_sampling_max_p = 0.4
        experiment_config.run.scheduled_sampling_ramp_epochs = 2
        training_model = trainers.ArOnsetTrainingModel(
            models.build_ar_onset_model(experiment_config),
            experiment_config=experiment_config,
        )
        callback = trainers.ScheduledSamplingRampCallback(
            training_model,
            max_p=0.4,
            ramp_epochs=2,
            warmup_epochs=1,
        )
        callback.on_epoch_begin(0)
        self.assertAlmostEqual(float(training_model.scheduled_sampling_p), 0.0)
        callback.on_epoch_begin(2)
        self.assertAlmostEqual(float(training_model.scheduled_sampling_p), 0.4)

    def test_train_step_runs_on_synthetic_batch(self) -> None:
        experiment_config = _tiny_experiment_config()
        times = np.asarray([0.05, 0.12], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(3, experiment_config.patch_input_dim()).astype(
                np.float32,
            ),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )
        batch = datasets.sample_to_training_batch(sample, experiment_config)
        tf_batch = {key: tf.constant(value) for key, value in batch.items()}

        base_model = models.build_ar_onset_model(experiment_config)
        training_model = trainers.ArOnsetTrainingModel(
            base_model,
            experiment_config=experiment_config,
        )
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )
        metrics = training_model.train_step((tf_batch,))
        self.assertIn("loss", metrics)
        self.assertIn(mn.AUX_F1_HUNGARIAN, metrics)
        self.assertNotIn("incremental_consistency_loss", metrics)

    def test_train_step_reports_incremental_consistency_loss(self) -> None:
        experiment_config = _tiny_experiment_config()
        experiment_config.run.lambda_incremental_consistency = 1.0
        experiment_config.run.incremental_consistency_max_steps = 4
        times = np.asarray([0.05, 0.12], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(3, experiment_config.patch_input_dim()).astype(
                np.float32,
            ),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )
        batch = datasets.sample_to_training_batch(sample, experiment_config)
        tf_batch = {key: tf.constant(value) for key, value in batch.items()}

        base_model = models.build_ar_onset_model(experiment_config)
        training_model = trainers.ArOnsetTrainingModel(
            base_model,
            experiment_config=experiment_config,
        )
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )
        metrics = training_model.train_step((tf_batch,))
        self.assertIn("incremental_consistency_loss", metrics)

    def test_test_step_reports_incremental_consistency_loss(self) -> None:
        experiment_config = _tiny_experiment_config()
        experiment_config.run.lambda_incremental_consistency = 1.0
        experiment_config.run.incremental_consistency_max_steps = 4
        times = np.asarray([0.05, 0.12], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(3, experiment_config.patch_input_dim()).astype(
                np.float32,
            ),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )
        batch = datasets.sample_to_training_batch(sample, experiment_config)
        tf_batch = {key: tf.constant(value) for key, value in batch.items()}

        base_model = models.build_ar_onset_model(experiment_config)
        training_model = trainers.ArOnsetTrainingModel(
            base_model,
            experiment_config=experiment_config,
        )
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )
        metrics = training_model.test_step((tf_batch,))
        self.assertIn("incremental_consistency_loss", metrics)
        self.assertGreater(float(metrics["incremental_consistency_loss"].numpy()), 0.0)

    def test_test_step_accumulates_metrics_across_validation_batches(self) -> None:
        experiment_config = _tiny_experiment_config()
        times = np.asarray([0.05, 0.12], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(3, experiment_config.patch_input_dim()).astype(
                np.float32,
            ),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )
        batch = datasets.sample_to_training_batch(sample, experiment_config)
        tf_batch = {key: tf.constant(value) for key, value in batch.items()}
        training_model = trainers.ArOnsetTrainingModel(
            models.build_ar_onset_model(experiment_config),
            experiment_config=experiment_config,
        )
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )

        training_model.test_step((tf_batch,))
        training_model.test_step((tf_batch,))

        self.assertEqual(float(training_model.loss_tracker.count.numpy()), 2.0)

    def test_test_step_metrics_match_base_model_with_incremental_loss(self) -> None:
        """Val metrics must use full-model forward (same path as offline debug)."""
        experiment_config = _tiny_experiment_config()
        experiment_config.run.overfit_one_song = True
        experiment_config.run.lambda_incremental_consistency = 1.0
        experiment_config.run.incremental_consistency_max_steps = 4
        times = np.asarray([0.05, 0.12, 0.20], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(3, experiment_config.patch_input_dim()).astype(
                np.float32,
            ),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )
        batch = datasets.sample_to_training_batch(sample, experiment_config)
        tf_batch = {key: tf.constant(value) for key, value in batch.items()}

        base_model = models.build_ar_onset_model(experiment_config)
        training_model = trainers.ArOnsetTrainingModel(
            base_model,
            experiment_config=experiment_config,
        )
        training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )
        val_metrics = training_model.test_step((tf_batch,))

        model_inputs = training_model._model_inputs(tf_batch)  # noqa: SLF001
        outputs = base_model(model_inputs, training=False)
        pred_times = losses.predicted_times_from_outputs(
            outputs["pointer_logits"],
            outputs["residual_sec"],
            patch_frames=experiment_config.model.patch_frames,
            hop_sec=experiment_config.dataset.hop_sec,
            use_soft_expected=False,
        )
        pred_np = pred_times.numpy()[0]
        mask = batch["onset_step_mask"][0] > 0.5
        pred_kept = pred_np[mask]
        target_kept = batch["target_times"][0][mask]
        n_matched, n_ref = timing_match.timing_match_counts_numpy(
            pred_kept,
            target_kept,
            tolerance_sec=experiment_config.run.tolerance_sec,
        )
        expected_ordered = timing_match.timing_match_rate_from_counts(
            n_matched,
            int(pred_kept.size),
            n_ref,
        )
        tp, fp, fn = trainers._ar_event_onset_counts_numpy(  # noqa: SLF001
            pred_np,
            batch["onset_step_mask"][0],
            batch["gt_times"][0],
            batch["gt_mask"][0],
            tolerance_sec=experiment_config.run.tolerance_sec,
        )
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        expected_f1 = 2.0 * precision * recall / (precision + recall + 1e-9)

        self.assertAlmostEqual(
            float(val_metrics[mn.TIMING_MATCH_TEACHER].numpy()),
            expected_ordered,
            places=5,
        )
        self.assertAlmostEqual(
            float(val_metrics[mn.AUX_F1_HUNGARIAN].numpy()),
            expected_f1,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
