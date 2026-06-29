"""Tests for AR tide iteration recipe diff helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ITER_PKG = Path(__file__).resolve().parents[1] / "scripts" / "ar_tide_iter"
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

import recipe_diff  # noqa: E402


class RecipeDiffTest(unittest.TestCase):
    def test_diff_blocks_ignores_artifact_paths(self) -> None:
        left = {"learning_rate": 1e-5, "model_output_dir": "a"}
        right = {"learning_rate": 2e-5, "model_output_dir": "b"}
        changes = recipe_diff.summarize_run_delta(right, baseline=left)
        self.assertEqual(list(changes), ["learning_rate"])
        self.assertEqual(changes["learning_rate"]["to"], 2e-5)

    def test_recipe_fingerprint_sorted(self) -> None:
        run = {
            "learning_rate": 1e-5,
            "lambda_residual": 10.0,
            "model_output_dir": "models_wsl/ar/tide_overfit_iter/iter40",
        }
        fp = recipe_diff.recipe_fingerprint(run)
        self.assertIn("lambda_residual=10.0", fp)
        self.assertNotIn("model_output_dir", fp)

    def test_collect_config_keys_unions_history(self) -> None:
        champion = {"run": {"epochs": 200}, "model": {"d_model": 256}}
        history = [{"run": {"learning_rate": 1e-5, "epochs": 200}}]
        keys = recipe_diff.collect_config_keys(champion=champion, configs=history)
        self.assertIn("learning_rate", keys["run"])
        self.assertIn("d_model", keys["model"])


if __name__ == "__main__":
    unittest.main()
