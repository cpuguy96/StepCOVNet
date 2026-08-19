"""Tests for scripts/eval_itgpt_placement.py helpers."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import eval_itgpt_placement  # noqa: E402

from stepcovnet.itgpt import config


class EvalItgptPlacementScriptTest(unittest.TestCase):
    def test_resolve_model_path_override_and_unique(self):
        experiment = config.ItgptExperimentConfig(
            dataset=config.ItgptDatasetConfig(training_index_path="x.json"),
        )
        self.assertEqual(
            eval_itgpt_placement._resolve_model_path(experiment, "ckpt.keras"),
            "ckpt.keras",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = pathlib.Path(tmpdir)
            (model_dir / "a.keras").write_bytes(b"x")
            experiment.run.model_output_dir = str(model_dir)
            resolved = eval_itgpt_placement._resolve_model_path(experiment, "")
            self.assertTrue(resolved.endswith("a.keras"))
            (model_dir / "best.keras").write_bytes(b"best")
            (model_dir / "last.keras").write_bytes(b"last")
            resolved = eval_itgpt_placement._resolve_model_path(experiment, "")
            self.assertTrue(resolved.endswith("best.keras"))
            (model_dir / "b.keras").write_bytes(b"y")
            (model_dir / "best.keras").unlink()
            (model_dir / "last.keras").unlink()
            with self.assertRaises(FileNotFoundError):
                eval_itgpt_placement._resolve_model_path(experiment, "")


if __name__ == "__main__":
    unittest.main()
