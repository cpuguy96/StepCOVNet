"""Tests for scripts/train_itgpt_placement.py helpers."""

from __future__ import annotations

import pathlib
import sys
import unittest

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import train_itgpt_placement  # noqa: E402


class TrainItgptPlacementScriptTest(unittest.TestCase):
    def test_parser_requires_config(self):
        parser = train_itgpt_placement._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            ["--config", "configs/ddc/itgpt_placement_fraxtil_exp_smoke.json"]
        )
        self.assertTrue(args.config.endswith("itgpt_placement_fraxtil_exp_smoke.json"))
        self.assertFalse(args.fresh)


if __name__ == "__main__":
    unittest.main()
