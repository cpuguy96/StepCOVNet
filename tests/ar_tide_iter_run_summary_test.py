"""Tests for autoresearch run summaries and run_overnight exit semantics."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "ar_tide_iter"))

import run_summary  # noqa: E402


class RunSummaryTest(unittest.TestCase):
    def _write_result(
        self,
        repo: pathlib.Path,
        *,
        row: dict,
        run: dict | None = None,
    ) -> None:
        config_dir = repo / "logs" / "ar_tide_iter" / "configs"
        config_dir.mkdir(parents=True)
        cfg = config_dir / f"{row['id']}.json"
        cfg.write_text(
            json.dumps({"run": run or {"learning_rate": 1e-4}}),
            encoding="utf-8",
        )
        row = dict(row)
        row.setdefault("config", str(cfg.relative_to(repo)).replace("\\", "/"))
        results = repo / "logs" / "ar_tide_iter" / "results.jsonl"
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def test_teacher_gate_remaps_exit_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            self._write_result(
                repo,
                row={
                    "id": "iter99",
                    "timestamp": "t",
                    "notes": "mem",
                    "model_path": "models_wsl/ar/tide_overfit_iter/iter99/ar_onset_model.keras",
                    "train_exit": 0,
                    "teacher": "535/634 (0.8438)",
                    "error": "teacher metrics not perfect",
                    "teacher_gate_failed": True,
                },
            )
            summary = run_summary.build_run_summary(
                "iter99",
                raw_exit_code=1,
                repo=repo,
            )
            self.assertEqual(summary["outcome"], run_summary.OUTCOME_TEACHER)
            self.assertEqual(summary["exit_code"], 0)
            self.assertEqual(summary["teacher_matched"], 535)

    def test_infra_failure_keeps_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            self._write_result(
                repo,
                row={
                    "id": "iter100",
                    "timestamp": "t",
                    "notes": "crash",
                    "model_path": "models_wsl/ar/tide_overfit_iter/iter100/ar_onset_model.keras",
                    "train_exit": 1,
                    "error": "checkpoint missing after train",
                },
            )
            summary = run_summary.build_run_summary(
                "iter100",
                raw_exit_code=1,
                repo=repo,
            )
            self.assertEqual(summary["outcome"], run_summary.OUTCOME_INFRA)
            self.assertEqual(summary["exit_code"], 1)

    def test_goal_passed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            self._write_result(
                repo,
                row={
                    "id": "iter101",
                    "timestamp": "t",
                    "notes": "win",
                    "model_path": "models_wsl/ar/tide_overfit_iter/iter101/ar_onset_model.keras",
                    "train_exit": 0,
                    "teacher": "634/634 (1.0000)",
                    "free_run": "634/634 (1.0000)",
                    "free_run_matched": 634,
                    "free_run_denom": 634,
                    "passed": True,
                },
            )
            summary = run_summary.build_run_summary(
                "iter101",
                raw_exit_code=0,
                repo=repo,
            )
            self.assertTrue(summary["goal_passed"])
            self.assertEqual(summary["outcome"], run_summary.OUTCOME_GOAL)


if __name__ == "__main__":
    unittest.main()
