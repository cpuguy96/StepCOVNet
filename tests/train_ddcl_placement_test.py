"""Tests for scripts/train_ddcl_placement.py helpers."""

from __future__ import annotations

import pathlib
import sys
import unittest

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import train_ddcl_placement  # noqa: E402


class TrainDdclPlacementScriptTest(unittest.TestCase):
    def test_parser_requires_config(self):
        parser = train_ddcl_placement._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args(
            ["--config", "configs/ddc/ddcl_placement_fraxtil_smoke.json"]
        )
        self.assertTrue(args.config.endswith("ddcl_placement_fraxtil_smoke.json"))
        self.assertFalse(args.fresh)
        fresh = parser.parse_args(
            [
                "--config",
                "configs/ddc/ddcl_placement_fraxtil_smoke.json",
                "--fresh",
            ]
        )
        self.assertTrue(fresh.fresh)


if __name__ == "__main__":
    unittest.main()
