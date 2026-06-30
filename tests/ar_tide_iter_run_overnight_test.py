"""Tests for run_overnight autoresearch CLI guards."""

from __future__ import annotations

import io
import pathlib
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ar_tide_iter"))

import run_overnight  # noqa: E402
from run_summary import SUMMARY_MARKER  # noqa: E402


class RunOvernightCliTest(unittest.TestCase):
    def test_hours_without_planner_is_blocked(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_overnight.py", "--hours", "1"],
        ):
            code = run_overnight.main()
        self.assertEqual(code, run_overnight.EXIT_MODE)

    def test_autoresearch_without_plan_exits_2(self) -> None:
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", ["run_overnight.py", "--autoresearch", "--once"]),
            patch.object(run_overnight, "NEXT_EXPERIMENT_PATH") as plan_path,
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            plan_path.is_file.return_value = False
            code = run_overnight.main()
        self.assertEqual(code, run_overnight.EXIT_NO_PLAN)
        self.assertIn(SUMMARY_MARKER, stdout.getvalue())
        self.assertIn("no_plan", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
