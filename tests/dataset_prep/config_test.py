"""Tests for dataset_prep.config."""

import json
import pathlib
import tempfile
import unittest

from stepcovnet.dataset_prep import config, constants


class PrepConfigTest(unittest.TestCase):
    def test_default_prep_config_matches_section_8(self):
        cfg = config.default_prep_config()
        self.assertEqual(cfg.input_dir, constants.DEFAULT_INPUT_DIR)
        self.assertEqual(cfg.output_dir, constants.DEFAULT_OUTPUT_DIR)
        self.assertEqual(cfg.export_mode, config.ExportMode.EXPORT_ALL_SINGLES)
        self.assertEqual(cfg.max_steps_per_chart, constants.MAX_STEPS_PER_CHART)
        self.assertFalse(cfg.export_legacy_txt)
        self.assertEqual(cfg.workers, 1)
        self.assertFalse(cfg.dry_run)
        self.assertFalse(cfg.overwrite)
        self.assertFalse(cfg.allow_over_cap)

    def test_validate_prep_config_accepts_defaults(self):
        config.validate_prep_config(config.default_prep_config())

    def test_validate_prep_config_rejects_zero_workers(self):
        cfg = config.PrepConfig(workers=0)
        with self.assertRaises(ValueError):
            config.validate_prep_config(cfg)

    def test_validate_prep_config_rejects_zero_step_cap(self):
        cfg = config.PrepConfig(max_steps_per_chart=0)
        with self.assertRaises(ValueError):
            config.validate_prep_config(cfg)

    def test_as_dict_and_from_dict_round_trip(self):
        cfg = config.PrepConfig(
            input_dir="data/raw_data/ITL Online 2026",
            workers=4,
            dry_run=True,
            allow_over_cap=True,
        )
        restored = config.PrepConfig.from_dict(cfg.as_dict())
        self.assertEqual(restored.input_dir, cfg.input_dir)
        self.assertEqual(restored.workers, 4)
        self.assertTrue(restored.dry_run)
        self.assertTrue(restored.allow_over_cap)

    def test_validate_prep_config_rejects_unsupported_export_mode(self):
        cfg = config.PrepConfig(export_mode="legacy_only")
        with self.assertRaises(ValueError):
            config.validate_prep_config(cfg)

    def test_save_and_load_prep_config_json(self):
        cfg = config.PrepConfig(output_dir="data/final_data", workers=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "prep.json"
            config.save_prep_config_json(cfg, str(path))
            with path.open(encoding="utf-8") as handle:
                raw = json.load(handle)
            self.assertEqual(raw["export_mode"], constants.EXPORT_MODE_ALL_SINGLES)
            loaded = config.load_prep_config_json(str(path))
        self.assertEqual(loaded.workers, 2)
        self.assertEqual(loaded.output_dir, "data/final_data")


if __name__ == "__main__":
    unittest.main()
