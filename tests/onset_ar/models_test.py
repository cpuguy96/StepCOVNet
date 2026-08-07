import dataclasses
import unittest

import numpy as np
import tensorflow as tf

from stepcovnet import onset_metric_names as mn
from stepcovnet import timing_match
from stepcovnet.onset_ar import (
    config,
    datasets,
    inference,
    models,
    targets,
    trainers,
)


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
        self.assertNotIn("gap_logits", outputs)

    def test_gap_residual_model_emits_gap_logits_without_pointer(self) -> None:
        experiment_config = _tiny_experiment_config()
        experiment_config.model.alignment = "gap_residual"
        experiment_config.model.gap_head = "dense"
        experiment_config.model.keep_absolute_pointer_head = False
        experiment_config.model.patch_delta_max_dense = 16
        experiment_config.model.patch_n_log_buckets = 4
        model = models.build_ar_onset_model(experiment_config)
        max_patches = experiment_config.max_encoder_patches()
        max_dec = experiment_config.max_decoder_len()
        patch_dim = experiment_config.patch_input_dim()
        gap_vocab_size = experiment_config.build_gap_vocab().vocab_size
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
        self.assertIn("gap_logits", outputs)
        self.assertNotIn("pointer_logits", outputs)
        self.assertEqual(outputs["gap_logits"].shape, (1, max_dec, gap_vocab_size))
        encoder, decoder = models.build_ar_onset_inference_models(
            model,
            experiment_config,
        )
        enc_out = encoder(
            {
                "mert_patches": inputs["mert_patches"],
                "patch_mask": inputs["patch_mask"],
            },
            training=False,
        )
        memory, _ = models.unpack_encoder_outputs(enc_out)
        dec_out = decoder(
            {
                "encoder_memory": memory,
                "patch_mask": inputs["patch_mask"],
                "decoder_input_ids": inputs["decoder_input_ids"],
                "decoder_mask": inputs["decoder_mask"],
            },
            training=False,
        )
        self.assertIn("gap_logits", dec_out)
        self.assertNotIn("pointer_logits", dec_out)

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

    def test_ordered_timing_match_is_published_on_multi_song_runs(self) -> None:
        """Hungarian F1 has a high chance floor, so the ordered metric must exist.

        Regression: it used to be created only for ``overfit_one_song`` runs, which
        left multi-song runs with no low-floor metric to select checkpoints on.
        """
        experiment_config = _tiny_experiment_config()
        experiment_config.run.overfit_one_song = False
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
        val_metrics = training_model.test_step((tf_batch,))
        self.assertIn(mn.TIMING_MATCH_TEACHER, val_metrics)
        self.assertIn(
            mn.val_name(mn.TIMING_MATCH_TEACHER),
            {mn.val_name(key) for key in val_metrics},
        )

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
        pred_times, _ = inference.decode_teacher_fed_times_tf(
            outputs,
            tf_batch,
            patch_frames=experiment_config.model.patch_frames,
            hop_sec=experiment_config.dataset.hop_sec,
            use_soft_expected=False,
            monotonic_pointer=True,
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


class PointerHeadTest(unittest.TestCase):
    """The pointer must score encoder content, not absolute patch indices."""

    def _inputs(
        self,
        experiment_config: config.ArExperimentConfig,
        *,
        n_patches: int,
        n_steps: int,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        patch_dim = experiment_config.patch_input_dim()
        return {
            "mert_patches": rng.standard_normal(
                (1, n_patches, patch_dim),
            ).astype("float32"),
            "patch_mask": np.ones((1, n_patches), dtype="float32"),
            "decoder_input_ids": rng.integers(
                0,
                experiment_config.build_vocab().vocab_size,
                (1, n_steps),
            ).astype("int32"),
            "decoder_mask": np.ones((1, n_steps), dtype="float32"),
        }

    def test_content_gap_depends_on_audio(self) -> None:
        experiment_config = _tiny_experiment_config()
        experiment_config.model.alignment = "gap_residual"
        experiment_config.model.gap_head = "content"
        experiment_config.model.keep_absolute_pointer_head = False
        experiment_config.model.patch_delta_max_dense = 16
        experiment_config.model.patch_n_log_buckets = 4
        model = models.build_ar_onset_model(experiment_config)
        inputs = self._inputs(experiment_config, n_patches=9, n_steps=5)
        inputs["prev_patch_indices"] = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int32)
        real = model(inputs, training=False)["gap_logits"].numpy()
        zeroed = model(
            {**inputs, "mert_patches": np.zeros_like(inputs["mert_patches"])},
            training=False,
        )["gap_logits"].numpy()
        self.assertGreater(float(np.abs(real - zeroed).max()), 1e-3)
        self.assertIn("prev_patch_indices", model.input)
        self.assertTrue(config.content_gap_active(experiment_config.model))
        encoder, decoder = models.build_ar_onset_inference_models(
            model,
            experiment_config,
        )
        enc_out = encoder(
            {
                "mert_patches": inputs["mert_patches"],
                "patch_mask": inputs["patch_mask"],
            },
            training=False,
        )
        memory, key_input = models.unpack_encoder_outputs(enc_out)
        dec_out = decoder(
            {
                "encoder_memory": memory,
                "pointer_key_input": key_input,
                "patch_mask": tf.constant(inputs["patch_mask"]),
                "decoder_input_ids": tf.constant(inputs["decoder_input_ids"]),
                "decoder_mask": tf.constant(inputs["decoder_mask"]),
                "prev_patch_indices": tf.constant(inputs["prev_patch_indices"]),
            },
            training=False,
        )
        self.assertIn("gap_logits", dec_out)
        self.assertNotIn("pointer_logits", dec_out)

    def test_content_gap_masks_oob_deltas(self) -> None:
        experiment_config = _tiny_experiment_config()
        experiment_config.model.alignment = "gap_residual"
        experiment_config.model.gap_head = "content"
        experiment_config.model.keep_absolute_pointer_head = False
        experiment_config.model.patch_delta_max_dense = 8
        experiment_config.model.patch_n_log_buckets = 2
        model = models.build_ar_onset_model(experiment_config)
        inputs = self._inputs(experiment_config, n_patches=4, n_steps=2)
        # prev=3 → Δ≥1 lands past the last valid patch.
        inputs["prev_patch_indices"] = np.asarray([[3, 3]], dtype=np.int32)
        logits = model(inputs, training=False)["gap_logits"].numpy()[0, 0]
        self.assertLess(float(logits[1]), -1e8)
        self.assertGreater(float(logits[0]), -1e8)

    def test_content_pointer_is_the_default(self) -> None:
        self.assertEqual(config.ArModelConfig().pointer_head, "content")
        self.assertTrue(config.content_pointer_active(config.ArModelConfig()))
        self.assertFalse(config.ArModelConfig().legacy_inverted_attention_masks)
        self.assertTrue(config.ArModelConfig().pointer_keys_pe_free)
        self.assertTrue(config.ArModelConfig().pointer_query_from_cross_attn)
        self.assertTrue(config.ArModelConfig().monotonic_pointer)
        self.assertFalse(config.ArModelConfig().decoder_cross_content_only)
        self.assertTrue(config.ArModelConfig().pointer_qk_layernorm)

    def test_decoder_cross_content_only_keeps_enc_pos_and_rebuilds(self) -> None:
        """Content-only cross must not prune enc_pos (inference needs memory)."""
        base = _tiny_experiment_config()
        experiment_config = config.ArExperimentConfig(
            dataset=base.dataset,
            model=dataclasses.replace(
                base.model,
                decoder_cross_content_only=True,
            ),
            run=base.run,
        )
        self.assertTrue(experiment_config.model.decoder_cross_content_only)
        model = models.build_ar_onset_model(experiment_config)
        names = {layer.name for layer in model.layers}
        self.assertIn("enc_pos", names)
        self.assertIn("cross_memory", names)
        self.assertNotIn("cross_memory_proj", names)
        encoder, decoder = models.build_ar_onset_inference_models(
            model,
            experiment_config,
        )
        inputs = self._inputs(experiment_config, n_patches=8, n_steps=3)
        enc_out = encoder(
            {
                "mert_patches": inputs["mert_patches"],
                "patch_mask": inputs["patch_mask"],
            },
            training=False,
        )
        memory, key_input = models.unpack_encoder_outputs(enc_out)
        out = decoder(
            {
                "encoder_memory": np.asarray(memory.numpy()),
                "pointer_key_input": np.asarray(key_input.numpy()),
                "patch_mask": inputs["patch_mask"],
                "decoder_input_ids": inputs["decoder_input_ids"],
                "decoder_mask": inputs["decoder_mask"],
            },
            training=False,
        )
        self.assertEqual(out["pointer_logits"].shape[-1], 8)

    def test_content_pointer_builds_cross_attn_query_and_pe_free_key(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        names = {layer.name for layer in model.layers}
        self.assertIn("pointer_cross_attn", names)
        self.assertIn("pointer_query", names)
        self.assertIn("pointer_key", names)
        self.assertIn("pointer_query_ln", names)
        self.assertIn("pointer_key_ln", names)
        self.assertIn("patch_embed", names)

    def test_unknown_pointer_head_is_rejected(self) -> None:
        model_config = config.ArModelConfig(pointer_head="dense")
        with self.assertRaises(ValueError):
            config.content_pointer_active(model_config)

    def test_content_pointer_logits_track_patch_count(self) -> None:
        """Logit count follows encoder length, so songs of any length work."""
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        for n_patches in (5, 11):
            inputs = self._inputs(experiment_config, n_patches=n_patches, n_steps=4)
            outputs = model(inputs, training=False)
            self.assertEqual(
                outputs["pointer_logits"].shape,
                (1, 4, n_patches),
            )

    def test_content_pointer_depends_on_audio(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        inputs = self._inputs(experiment_config, n_patches=9, n_steps=5)
        real = model(inputs, training=False)["pointer_logits"].numpy()
        zeroed = model(
            {**inputs, "mert_patches": np.zeros_like(inputs["mert_patches"])},
            training=False,
        )["pointer_logits"].numpy()
        self.assertGreater(float(np.abs(real - zeroed).max()), 1e-3)

    def test_pointer_query_reads_pe_free_patch_embed_at_init(self) -> None:
        """Pointer queries must depend on ``patch_embed``, not PE-only memory.

        Shuffle is a weak init probe (near-uniform attention is permutation
        invariant). Zeroing patches must move ``pointer_query``.
        """
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        inputs = self._inputs(experiment_config, n_patches=12, n_steps=6)
        extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("pointer_query").output,
        )
        q_matched = extractor(inputs, training=False).numpy()
        q_zero = extractor(
            {**inputs, "mert_patches": np.zeros_like(inputs["mert_patches"])},
            training=False,
        ).numpy()
        self.assertGreater(float(np.linalg.norm(q_matched - q_zero)), 1e-3)

    def test_pe_free_pointer_keys_are_contextualized(self) -> None:
        """Pe-free keys must be encoder outputs, not raw ``Dense(MERT)``.

        Encoding with absolute PE first made ``memory`` shuffle-invariant; the
        pe-free workaround keyed on ``patch_embed`` and skipped the encoder.
        Encode-then-PE keeps keys contextualized and PE-free.
        """
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        inputs = self._inputs(experiment_config, n_patches=12, n_steps=4)
        n_enc = experiment_config.model.n_enc_layers
        extractor = tf.keras.Model(
            inputs=model.input,
            outputs={
                "raw": model.get_layer("patch_embed").output,
                "content": model.get_layer(f"enc_{n_enc - 1}_ln2").output,
                "memory": model.get_layer("enc_pos").output,
                "keys": model.get_layer("pointer_key").output,
            },
        )
        out = extractor(inputs, training=False)
        raw = out["raw"].numpy()
        content = out["content"].numpy()
        memory = out["memory"].numpy()
        keys = out["keys"].numpy()
        self.assertGreater(float(np.linalg.norm(content - raw)), 1e-3)
        self.assertGreater(float(np.linalg.norm(memory - content)), 1e-3)
        # Pointer keys are a projection of contextualized content, not raw embed.
        self.assertGreater(
            float(np.linalg.norm(keys - raw[:, : keys.shape[1], : keys.shape[2]])),
            1e-3,
        )

    def test_legacy_index_pointer_still_builds(self) -> None:
        base = _tiny_experiment_config()
        experiment_config = config.ArExperimentConfig(
            dataset=base.dataset,
            model=dataclasses.replace(base.model, pointer_head="index"),
            run=base.run,
        )
        self.assertFalse(config.content_pointer_active(experiment_config.model))
        model = models.build_ar_onset_model(experiment_config)
        layer_names = {layer.name for layer in model.layers}
        self.assertIn("pointer_logits", layer_names)
        self.assertNotIn("pointer_logits_content", layer_names)

    def test_content_pointer_size_is_independent_of_max_patches(self) -> None:
        """Index-head size grows with the padded patch axis; content-head does not.

        This is what lets one model serve songs of any length, and why the head
        stops costing ~1.4M params at production ``max_patches``.
        """
        base = _tiny_experiment_config()
        counts: dict[tuple[str, float], int] = {}
        for head in ("content", "index"):
            for max_audio_seconds in (1.0, 8.0):
                experiment_config = config.ArExperimentConfig(
                    dataset=dataclasses.replace(
                        base.dataset,
                        max_audio_seconds=max_audio_seconds,
                    ),
                    model=dataclasses.replace(base.model, pointer_head=head),
                    run=base.run,
                )
                counts[head, max_audio_seconds] = models.build_ar_onset_model(
                    experiment_config,
                ).count_params()
        self.assertEqual(counts["content", 1.0], counts["content", 8.0])
        self.assertLess(counts["index", 1.0], counts["index", 8.0])
        # Index head grows with the padded patch axis; content head does not.
        self.assertGreater(
            counts["index", 8.0] - counts["index", 1.0],
            counts["content", 8.0] - counts["content", 1.0],
        )

    def test_inference_rebuild_matches_full_model_for_both_heads(self) -> None:
        """Old checkpoints keep rebuilding, and the new head rebuilds too."""
        base = _tiny_experiment_config()
        for head in ("content", "index"):
            with self.subTest(head=head):
                experiment_config = config.ArExperimentConfig(
                    dataset=base.dataset,
                    model=dataclasses.replace(base.model, pointer_head=head),
                    run=base.run,
                )
                model = models.build_ar_onset_model(experiment_config)
                encoder, decoder = models.build_ar_onset_inference_models(
                    model,
                    experiment_config,
                )
                inputs = self._inputs(experiment_config, n_patches=7, n_steps=4)
                expected = model(inputs, training=False)["pointer_logits"].numpy()
                enc_out = encoder(
                    {
                        "mert_patches": inputs["mert_patches"],
                        "patch_mask": inputs["patch_mask"],
                    },
                )
                memory, key_input = models.unpack_encoder_outputs(enc_out)
                decoder_inputs = {
                    "encoder_memory": np.asarray(memory),
                    "patch_mask": inputs["patch_mask"],
                    "decoder_input_ids": inputs["decoder_input_ids"],
                    "decoder_mask": inputs["decoder_mask"],
                }
                if head == "content":
                    decoder_inputs["pointer_key_input"] = np.asarray(key_input)
                actual = decoder(decoder_inputs)["pointer_logits"].numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_content_pointer_masks_padded_patches(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        inputs = self._inputs(experiment_config, n_patches=8, n_steps=4)
        inputs["patch_mask"] = np.asarray(
            [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype="float32",
        )
        logits = model(inputs, training=False)["pointer_logits"].numpy()
        self.assertTrue(bool(np.all(logits[0, :, 3:] < -1e8)))
        self.assertTrue(bool(np.all(logits[0, :, :3] > -1e8)))


if __name__ == "__main__":
    unittest.main()
