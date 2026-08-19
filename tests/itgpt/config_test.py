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
            ),
            model=config.ItgptModelConfig(d_model=32, n_heads=4, n_enc_layers=1),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "cfg.json"
            path.write_text(json.dumps(experiment.as_dict()), encoding="utf-8")
            loaded = config.ItgptExperimentConfig.from_json(path)
        self.assertEqual(loaded.dataset.max_beats, 64)
        self.assertEqual(loaded.model.d_model, 32)
        self.assertEqual(loaded.model.n_heads, 4)
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


if __name__ == "__main__":
    unittest.main()
