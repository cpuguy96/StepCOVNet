import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, datasets, inference, models, targets


def _tiny_experiment_config() -> config.ArExperimentConfig:
    return config.ArExperimentConfig(
        dataset=config.ArDatasetConfig(max_audio_seconds=1.0, hop_sec=0.01),
        model=config.ArModelConfig(
            patch_frames=4,
            d_model=32,
            n_enc_layers=1,
            n_dec_layers=2,
            num_heads=2,
            max_decode_steps=16,
            delta_max_dense=8,
            n_log_buckets=4,
            n_first_abs_bins=8,
            dropout_rate=0.0,
        ),
        run=config.ArRunConfig(epochs=1, model_output_dir=""),
    )


def _synthetic_batch(experiment_config: config.ArExperimentConfig):
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
    return datasets.sample_to_training_batch(sample, experiment_config)


class InferenceDecodeTest(unittest.TestCase):
    def test_build_decoder_inputs_for_onset_tokens(self) -> None:
        dec_in, dec_mask = inference.build_decoder_inputs_for_onset_tokens(
            np.asarray([10, 11, 12], dtype=np.int32),
            max_decoder_len=8,
        )
        np.testing.assert_array_equal(dec_in[0, :4], [targets.BOS_ID, 10, 11, 12])
        np.testing.assert_array_equal(dec_mask[0, :4], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(dec_in[0, 4:], 0)
        np.testing.assert_array_equal(dec_mask[0, 4:], 0.0)

    def test_gt_parallel_matches_teacher_fed_times(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        batch = _synthetic_batch(experiment_config)
        batch_tf = {key: tf.constant(value) for key, value in batch.items()}
        outputs = model(
            {
                "mert_patches": batch_tf["mert_patches"],
                "patch_mask": batch_tf["patch_mask"],
                "decoder_input_ids": batch_tf["decoder_input_ids"],
                "decoder_mask": batch_tf["decoder_mask"],
            },
            training=False,
        )
        teacher_times = inference.decode_teacher_fed_times_numpy(
            outputs["pointer_logits"].numpy(),
            outputs["residual_sec"].numpy(),
            batch["onset_step_mask"][0],
            patch_frames=experiment_config.model.patch_frames,
            hop_sec=experiment_config.dataset.hop_sec,
        )
        onset_mask = batch["onset_step_mask"][0] > 0.5
        gt_tokens = batch["decoder_target_ids"][0][onset_mask]
        parallel_times = inference.decode_parallel_pointer_times_numpy(
            model,
            batch["mert_patches"],
            batch["patch_mask"],
            gt_tokens,
            experiment_config=experiment_config,
        )
        np.testing.assert_allclose(parallel_times, teacher_times, rtol=1e-5, atol=1e-5)

    def test_gate_decode_matches_two_pass(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        batch = _synthetic_batch(experiment_config)
        decode_kwargs = {
            "max_decoder_len": experiment_config.max_decoder_len(),
            "patch_frames": experiment_config.model.patch_frames,
            "hop_sec": experiment_config.dataset.hop_sec,
            "experiment_config": experiment_config,
        }
        gate = inference.decode_autoregressive_gate_with_stats_numpy(
            model,
            batch["mert_patches"],
            batch["patch_mask"],
            **decode_kwargs,
        )
        two_pass = inference.decode_autoregressive_two_pass_with_stats_numpy(
            model,
            batch["mert_patches"],
            batch["patch_mask"],
            **decode_kwargs,
        )
        np.testing.assert_allclose(gate.times, two_pass.times, rtol=1e-5, atol=1e-5)
        self.assertEqual(gate.n_forward_steps, two_pass.n_forward_steps)

    def test_two_pass_reuses_incremental_tokens(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        batch = _synthetic_batch(experiment_config)
        decode_kwargs = {
            "max_decoder_len": experiment_config.max_decoder_len(),
            "patch_frames": experiment_config.model.patch_frames,
            "hop_sec": experiment_config.dataset.hop_sec,
            "experiment_config": experiment_config,
        }
        incremental = inference.decode_autoregressive_with_stats_numpy(
            model,
            batch["mert_patches"],
            batch["patch_mask"],
            use_kv_cache=True,
            **decode_kwargs,
        )
        two_pass = inference.decode_autoregressive_two_pass_with_stats_numpy(
            model,
            batch["mert_patches"],
            batch["patch_mask"],
            use_kv_cache=True,
            token_pass=incremental,
            **decode_kwargs,
        )
        self.assertIsNotNone(incremental.onset_token_ids)
        np.testing.assert_array_equal(
            incremental.onset_token_ids,
            two_pass.onset_token_ids,
        )
        self.assertEqual(
            two_pass.n_forward_steps,
            incremental.n_forward_steps + 1,
        )

    def test_encoder_memory_cache_reused(self) -> None:
        experiment_config = _tiny_experiment_config()
        model = models.build_ar_onset_model(experiment_config)
        batch = _synthetic_batch(experiment_config)
        inference.clear_encoder_memory_cache(model)
        memory_a, _ = inference.get_encoder_memory_numpy(
            model,
            batch["mert_patches"],
            batch["patch_mask"],
            experiment_config,
        )
        memory_b, _ = inference.get_encoder_memory_numpy(
            model,
            batch["mert_patches"],
            batch["patch_mask"],
            experiment_config,
        )
        np.testing.assert_array_equal(memory_a, memory_b)
        self.assertIs(memory_a, memory_b)


if __name__ == "__main__":
    unittest.main()
