import unittest

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, datasets, models, targets, trainers


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


class TrainersTest(unittest.TestCase):
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
        self.assertIn("event_onset_f1", metrics)


if __name__ == "__main__":
    unittest.main()
