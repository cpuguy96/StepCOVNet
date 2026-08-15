"""Tests for DDC placement configs."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from stepcovnet.ddc import config


class DdcConfigTest(unittest.TestCase):
    def test_json_round_trip(self):
        experiment = config.PlacementExperimentConfig(
            dataset=config.PlacementDatasetConfig(
                training_index_path="data/literature_fraxtil_orig/training_index_standard.json",
                batch_size=8,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "cfg.json"
            path.write_text(json.dumps(experiment.as_dict()), encoding="utf-8")
            loaded = config.PlacementExperimentConfig.from_json(path)
        self.assertEqual(loaded.dataset.batch_size, 8)
        self.assertEqual(loaded.model.lstm_layers, 2)

    def test_repo_configs_load(self):
        smoke = config.PlacementExperimentConfig.from_json(
            "configs/ddc/placement_fraxtil_smoke.json"
        )
        full = config.PlacementExperimentConfig.from_json(
            "configs/ddc/placement_fraxtil.json"
        )
        self.assertEqual(smoke.dataset.max_train_songs, 2)
        self.assertEqual(full.model.lstm_units, 200)

    def test_dataset_rejects_bad_batch(self):
        with self.assertRaises(ValueError):
            config.PlacementDatasetConfig(training_index_path="x", batch_size=0)
        with self.assertRaises(ValueError):
            config.PlacementDatasetConfig(training_index_path="x", nunroll=0)

    def test_model_and_run_validation(self):
        with self.assertRaises(ValueError):
            config.PlacementModelConfig(lstm_units=0)
        with self.assertRaises(ValueError):
            config.PlacementModelConfig(lstm_layers=0)
        with self.assertRaises(ValueError):
            config.PlacementModelConfig(dropout_rate=1.0)
        with self.assertRaises(ValueError):
            config.PlacementModelConfig(dnn_sizes=[])
        with self.assertRaises(ValueError):
            config.PlacementModelConfig(dnn_sizes=[0])
        with self.assertRaises(ValueError):
            config.PlacementRunConfig(epoch=0)
        with self.assertRaises(ValueError):
            config.PlacementRunConfig(learning_rate=0.0)


if __name__ == "__main__":
    unittest.main()
