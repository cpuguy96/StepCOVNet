"""Tests for the ITGPT placement Keras model."""

from __future__ import annotations

import pathlib
import tempfile
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

    def test_mert_forward_train_and_keras_save_reload_round_trip(self):
        trainers.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        feature_dim = 16
        model = trainers.compile_placement_model(
            models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                n_enc_layers=1,
                dropout_rate=0.1,
                max_beats=n_beats,
                audio_feature_type="mert",
                n_frames_per_beat=constants.CNN_FRAMES_OUT,
                feature_dim=feature_dim,
                model_name="itgpt_mert_test",
            ),
            config.ItgptRunConfig(
                learning_rate=1e-4,
                weight_decay=0.0,
                clipnorm=1.0,
            ),
        )
        rng = np.random.default_rng(7)
        inputs = {
            "audio": rng.standard_normal(
                (1, n_beats, constants.CNN_FRAMES_OUT, feature_dim)
            ).astype(np.float32),
            "bpm": np.array([[140.0]], dtype=np.float32),
            "difficulty": np.array([[8.0]], dtype=np.float32),
        }
        labels = np.zeros((1, n_beats, constants.N_SLOTS), dtype=np.float32)
        labels[0, 0, 0] = 1.0
        pred_before = np.asarray(model(inputs, training=False))
        self.assertEqual(pred_before.shape, (1, n_beats, constants.N_SLOTS))
        loss = model.train_on_batch(inputs, labels)
        self.assertTrue(np.isfinite(loss))
        pred_trained = np.asarray(model(inputs, training=False))
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = pathlib.Path(tmpdir) / "mert_roundtrip.keras"
            model.save(ckpt)
            reloaded = trainers.load_placement_model(ckpt)
            pred_reloaded = np.asarray(reloaded(inputs, training=False))
        np.testing.assert_allclose(pred_trained, pred_reloaded, atol=1e-5)
        with self.assertRaises(ValueError):
            models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                audio_feature_type="mert",
                n_frames_per_beat=16,
            )

    def test_early_film_identity_init_train_and_keras_reload(self):
        trainers.set_seed(0)
        n_beats = constants.CHUNK_ALIGN
        feature_dim = 16
        film_layer = models.ItgptConditionFilm(d_model=32)
        rng = np.random.default_rng(11)
        subbeat = rng.standard_normal(
            (n_beats, constants.CNN_FRAMES_OUT, 32)
        ).astype(np.float32)
        cond = rng.standard_normal((1, 64)).astype(np.float32)
        subbeat_out = np.asarray(
            film_layer(
                subbeat,
                cond,
                batch_size=1,
                n_beats=n_beats,
                frames_per_beat=constants.CNN_FRAMES_OUT,
            )
        )
        np.testing.assert_allclose(subbeat_out, subbeat, atol=1e-6)

        model = trainers.compile_placement_model(
            models.build_itgpt_placement_model(
                d_model=32,
                n_heads=4,
                n_enc_layers=1,
                dropout_rate=0.0,
                max_beats=n_beats,
                audio_feature_type="mert",
                n_frames_per_beat=constants.CNN_FRAMES_OUT,
                feature_dim=feature_dim,
                early_film=True,
                model_name="itgpt_mert_film_test",
            ),
            config.ItgptRunConfig(
                learning_rate=1e-3,
                weight_decay=0.0,
                clipnorm=1.0,
            ),
        )
        inputs = {
            "audio": rng.standard_normal(
                (1, n_beats, constants.CNN_FRAMES_OUT, feature_dim)
            ).astype(np.float32),
            "bpm": np.array([[140.0]], dtype=np.float32),
            "difficulty": np.array([[8.0]], dtype=np.float32),
        }
        labels = np.zeros((1, n_beats, constants.N_SLOTS), dtype=np.float32)
        labels[0, 0, 0] = 1.0
        loss = model.train_on_batch(inputs, labels)
        self.assertTrue(np.isfinite(loss))
        pred_trained = np.asarray(model(inputs, training=False))
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = pathlib.Path(tmpdir) / "mert_film_roundtrip.keras"
            model.save(ckpt)
            reloaded = trainers.load_placement_model(ckpt)
            self.assertTrue(reloaded.early_film)
            pred_reloaded = np.asarray(reloaded(inputs, training=False))
        np.testing.assert_allclose(pred_trained, pred_reloaded, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
