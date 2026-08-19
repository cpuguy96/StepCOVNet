"""Tests for the ITGPT placement Keras model."""

from __future__ import annotations

import unittest

import numpy as np

from stepcovnet.itgpt import config, constants, models, trainers


class ItgptModelsTest(unittest.TestCase):
    def test_forward_shape_and_finite_train_step(self):
        trainers.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        model = trainers.compile_placement_model(
            models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                n_enc_layers=1,
                cnn_hidden=8,
                dropout_rate=0.0,
                max_beats=n_beats,
                model_name="itgpt_shape_test",
            ),
            config.ItgptRunConfig(
                learning_rate=1e-4,
                clipnorm=1.0,
            ),
        )
        inputs = {
            "audio": np.zeros(
                (
                    1,
                    n_beats,
                    constants.N_FRAMES_PER_BEAT,
                    constants.N_MELS,
                    constants.N_CHANNELS,
                ),
                dtype=np.float32,
            ),
            "bpm": np.array([[140.0]], dtype=np.float32),
            "difficulty": np.array([[8.0]], dtype=np.float32),
        }
        labels = np.zeros((1, n_beats, constants.N_SLOTS), dtype=np.float32)
        labels[0, 0, 0] = 1.0
        pred = model(inputs, training=False)
        self.assertEqual(tuple(pred.shape), (1, n_beats, constants.N_SLOTS))
        loss = model.train_on_batch(inputs, labels)
        self.assertTrue(np.isfinite(loss))
        weights = models.grid_importance_weights()
        self.assertEqual(tuple(weights.shape), (constants.N_SLOTS,))
        self.assertEqual(float(weights[0]), constants.GRID_WEIGHT_16TH)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(d_model=32, n_heads=3)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(n_enc_layers=0)


if __name__ == "__main__":
    unittest.main()
