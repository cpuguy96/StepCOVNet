"""Tests for ITGPT placement JSON config."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from stepcovnet.itgpt import config


class ItgptConfigTest(unittest.TestCase):
    def test_round_trip_json(self):
        experiment = config.ItgptExperimentConfig(
            dataset=config.ItgptDatasetConfig(
                training_index_path="data/literature_fraxtil_exp/training_index_standard.json",
                max_beats=64,
                exclude_normalized_ids=["bad_song"],
                audio_feature_type="mert",
                n_frames_per_beat=24,
                mert_cache_suffix=".mert_l2_6_10.npy",
                mert_spectral_flux=True,
            ),
            model=config.ItgptModelConfig(
                d_model=32,
                n_heads=4,
                n_enc_layers=1,
                early_film=True,
                cyclic_measure_pos=True,
                continuous_beat_conv=True,
            ),
            run=config.ItgptRunConfig(
                epoch=10,
                weight_decay=0.0,
                eval_slot_f1_every=16,
                lr_schedule="cosine",
                warmup_epochs=2,
                min_learning_rate=1e-6,
                focal_gamma=1.0,
                grid_weight_24th=2.0,
                gradient_accumulation_steps=4,
                use_ema=True,
                ema_momentum=0.995,
                max_bar_shift=15,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "cfg.json"
            path.write_text(json.dumps(experiment.as_dict()), encoding="utf-8")
            loaded = config.ItgptExperimentConfig.from_json(path)
        self.assertEqual(loaded.dataset.max_beats, 64)
        self.assertEqual(loaded.dataset.exclude_normalized_ids, ["bad_song"])
        self.assertEqual(loaded.dataset.audio_feature_type, "mert")
        self.assertEqual(loaded.dataset.n_frames_per_beat, 24)
        self.assertEqual(loaded.dataset.mert_cache_suffix, ".mert_l2_6_10.npy")
        self.assertTrue(loaded.dataset.mert_spectral_flux)
        self.assertFalse(
            config.ItgptDatasetConfig(training_index_path="x.json").mert_spectral_flux
        )
        self.assertEqual(loaded.model.d_model, 32)
        self.assertEqual(loaded.model.n_heads, 4)
        self.assertTrue(loaded.model.early_film)
        self.assertTrue(loaded.model.cyclic_measure_pos)
        self.assertTrue(loaded.model.continuous_beat_conv)
        self.assertFalse(config.ItgptModelConfig().early_film)
        self.assertFalse(config.ItgptModelConfig().cyclic_measure_pos)
        self.assertFalse(config.ItgptModelConfig().continuous_beat_conv)
        self.assertEqual(loaded.run.weight_decay, 0.0)
        self.assertEqual(loaded.run.eval_slot_f1_every, 16)
        self.assertEqual(loaded.run.lr_schedule, "cosine")
        self.assertEqual(loaded.run.warmup_epochs, 2)
        self.assertAlmostEqual(loaded.run.min_learning_rate, 1e-6)
        self.assertAlmostEqual(loaded.run.focal_gamma, 1.0)
        self.assertAlmostEqual(loaded.run.grid_weight_24th, 2.0)
        self.assertEqual(loaded.run.gradient_accumulation_steps, 4)
        self.assertEqual(config.ItgptRunConfig().gradient_accumulation_steps, 1)
        self.assertTrue(loaded.run.use_ema)
        self.assertAlmostEqual(loaded.run.ema_momentum, 0.995)
        self.assertFalse(config.ItgptRunConfig().use_ema)
        self.assertEqual(loaded.run.max_bar_shift, 15)
        self.assertEqual(config.ItgptRunConfig().max_bar_shift, 0)
        self.assertFalse(loaded.run.mixed_precision)
        self.assertFalse(loaded.run.jit_compile)

    def test_rejects_heads_not_dividing_width(self):
        with self.assertRaises(ValueError):
            config.ItgptModelConfig(d_model=32, n_heads=3)

    def test_rejects_invalid_sizes(self):
        with self.assertRaises(ValueError):
            config.ItgptDatasetConfig(training_index_path="x.json", batch_size=0)
        with self.assertRaises(ValueError):
            config.ItgptDatasetConfig(training_index_path="x.json", max_beats=32)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(epoch=0)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(learning_rate=0.0)
        with self.assertRaises(ValueError):
            config.ItgptModelConfig(d_model=0)
        with self.assertRaises(ValueError):
            config.ItgptModelConfig(n_enc_layers=0)
        with self.assertRaises(ValueError):
            config.ItgptModelConfig(cnn_hidden=0)
        with self.assertRaises(ValueError):
            config.ItgptModelConfig(dropout_rate=1.0)
        with self.assertRaises(ValueError):
            config.ItgptDatasetConfig(
                training_index_path="x.json", audio_feature_type="invalid"
            )
        with self.assertRaises(ValueError):
            config.ItgptDatasetConfig(
                training_index_path="x.json",
                audio_feature_type="mert",
                n_frames_per_beat=16,
            )
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(weight_decay=-0.1)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(eval_slot_f1_every=-1)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(lr_schedule="linear")
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(warmup_epochs=-1)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(epoch=4, lr_schedule="cosine", warmup_epochs=4)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(min_learning_rate=-1e-6)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(learning_rate=1e-4, min_learning_rate=2e-4)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(focal_gamma=-0.5)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(grid_weight_24th=0.0)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(gradient_accumulation_steps=0)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(ema_momentum=1.0)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(ema_momentum=0.0)
        with self.assertRaises(ValueError):
            config.ItgptRunConfig(max_bar_shift=-1)
        with self.assertRaises(ValueError):
            config.ItgptDatasetConfig(
                training_index_path="x.json", mert_cache_suffix="bad_suffix"
            )


if __name__ == "__main__":
    unittest.main()
