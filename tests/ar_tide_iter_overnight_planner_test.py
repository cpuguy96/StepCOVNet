"""Tests for history-driven overnight scratch experiment planner."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ITER_PKG = Path(__file__).resolve().parents[1] / "scripts" / "ar_tide_iter"
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

import overnight_planner  # noqa: E402


def _write_cfg(repo: Path, exp_id: str, run: dict) -> str:
    cfg_path = repo / "logs" / "ar_tide_iter" / "configs" / f"{exp_id}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"run": run}), encoding="utf-8")
    return str(cfg_path.relative_to(repo)).replace("\\", "/")


class OvernightPlannerTest(unittest.TestCase):
    def test_plan_parents_from_best_teacher_not_last(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            log_dir = repo / "logs" / "ar_tide_iter"
            log_dir.mkdir(parents=True)

            cfg43 = _write_cfg(
                repo,
                "iter43",
                {"learning_rate": 2e-5, "scheduled_sampling_max_p": 0.2},
            )
            cfg44 = _write_cfg(
                repo,
                "iter44",
                {"learning_rate": 5e-5, "scheduled_sampling_max_p": 0.0},
            )
            rows = [
                {
                    "id": "iter43",
                    "timestamp": "t1",
                    "notes": "Scratch train low teacher",
                    "model_path": "m43",
                    "config": cfg43,
                    "train_exit": 0,
                    "teacher": "56/634 (0.0883)",
                    "teacher_gate_failed": True,
                },
                {
                    "id": "iter44",
                    "timestamp": "t2",
                    "notes": "Scratch memorization better teacher",
                    "model_path": "m44",
                    "config": cfg44,
                    "train_exit": 0,
                    "teacher": "257/634 (0.4054)",
                    "teacher_gate_failed": True,
                },
            ]
            (log_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            plan = overnight_planner.plan_next_experiment(repo=repo)
            assert plan is not None
            self.assertEqual(plan["id"], "iter45")
            self.assertIn("iter44", plan["notes"])

    def test_plan_explores_untried_value_from_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            log_dir = repo / "logs" / "ar_tide_iter"
            log_dir.mkdir(parents=True)

            cfg_a = _write_cfg(repo, "iter43", {"learning_rate": 5e-5})
            cfg_b = _write_cfg(repo, "iter44", {"learning_rate": 1e-4})
            rows = [
                {
                    "id": "iter43",
                    "timestamp": "t1",
                    "notes": "Scratch a",
                    "model_path": "m1",
                    "config": cfg_a,
                    "train_exit": 0,
                    "teacher": "100/634 (0.1577)",
                    "teacher_gate_failed": True,
                },
                {
                    "id": "iter44",
                    "timestamp": "t2",
                    "notes": "Scratch b",
                    "model_path": "m2",
                    "config": cfg_b,
                    "train_exit": 0,
                    "teacher": "200/634 (0.3155)",
                    "teacher_gate_failed": True,
                },
            ]
            (log_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            plan = overnight_planner.plan_next_experiment(repo=repo)
            assert plan is not None
            self.assertEqual(plan["id"], "iter45")
            run = plan.get("run", {})
            self.assertIn("learning_rate", run)
            self.assertIn(run["learning_rate"], {5e-5, 1e-4, 2e-5})

    def test_plan_pins_epochs_to_champion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            log_dir = repo / "logs" / "ar_tide_iter"
            log_dir.mkdir(parents=True)
            cfg44 = _write_cfg(repo, "iter44", {"learning_rate": 5e-5, "epochs": 200})
            cfg_old = _write_cfg(repo, "iter01", {"learning_rate": 5e-5, "epochs": 150})
            rows = [
                {
                    "id": "iter01",
                    "timestamp": "t0",
                    "notes": "warm",
                    "model_path": "m1",
                    "config": cfg_old,
                    "train_exit": 0,
                    "free_run": "614/634 (0.9685)",
                    "free_run_matched": 614,
                    "free_run_denom": 634,
                },
                {
                    "id": "iter44",
                    "timestamp": "t1",
                    "notes": "Scratch memorization",
                    "model_path": "m44",
                    "config": cfg44,
                    "train_exit": 0,
                    "teacher": "257/634 (0.4054)",
                    "teacher_gate_failed": True,
                },
            ]
            (log_dir / "results.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            plan = overnight_planner.plan_next_experiment(repo=repo)
            assert plan is not None
            self.assertEqual(plan.get("run", {}).get("epochs"), 200)

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            log_dir = repo / "logs" / "ar_tide_iter"
            log_dir.mkdir(parents=True)
            row = {
                "id": "iter99",
                "attempt": 1,
                "timestamp": "t",
                "notes": "n",
                "model_path": "m",
                "config": "c.json",
                "train_exit": 0,
                "free_run": "634/634 (1.0000)",
                "free_run_matched": 634,
                "free_run_denom": 634,
                "passed": True,
            }
            (log_dir / "results.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            self.assertIsNone(overnight_planner.plan_next_experiment(repo=repo))


if __name__ == "__main__":
    unittest.main()
