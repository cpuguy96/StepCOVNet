"""Tests for DDCL placement JSON config."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from stepcovnet.ddcl import config


class DdclConfigTest(unittest.TestCase):
    def test_round_trip_json(self):
        experiment = config.DdclExperimentConfig(
            dataset=config.DdclDatasetConfig(
                training_index_path="data/literature_fraxtil_orig/training_index_standard.json",
                memlen=3,
                batch_size=4,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "cfg.json"
            path.write_text(json.dumps(experiment.as_dict()), encoding="utf-8")
            loaded = config.DdclExperimentConfig.from_json(path)
        self.assertEqual(loaded.dataset.memlen, 3)
        self.assertEqual(loaded.dataset.batch_size, 4)
        self.assertEqual(loaded.model.lstm_units, 200)

    def test_rejects_bad_batch_size(self):
        with self.assertRaises(ValueError):
            config.DdclDatasetConfig(training_index_path="x.json", batch_size=0)


if __name__ == "__main__":
    unittest.main()
