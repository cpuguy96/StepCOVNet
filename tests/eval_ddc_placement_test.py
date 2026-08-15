"""Tests for scripts/eval_ddc_placement.py helpers."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import eval_ddc_placement  # noqa: E402

from stepcovnet.ddc import config


class EvalDdcPlacementScriptTest(unittest.TestCase):
    def test_resolve_model_path_override_and_unique(self):
        experiment = config.PlacementExperimentConfig(
            dataset=config.PlacementDatasetConfig(training_index_path="x.json"),
        )
        self.assertEqual(
            eval_ddc_placement._resolve_model_path(experiment, "ckpt.keras"),
            "ckpt.keras",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = pathlib.Path(tmpdir)
            (model_dir / "a.keras").write_bytes(b"x")
            experiment.run.model_output_dir = str(model_dir)
            resolved = eval_ddc_placement._resolve_model_path(experiment, "")
            self.assertTrue(resolved.endswith("a.keras"))
            (model_dir / "b.keras").write_bytes(b"y")
            with self.assertRaises(FileNotFoundError):
                eval_ddc_placement._resolve_model_path(experiment, "")


if __name__ == "__main__":
    unittest.main()
