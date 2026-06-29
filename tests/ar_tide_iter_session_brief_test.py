import json
import pathlib
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class ArTideIterSessionBriefTest(unittest.TestCase):
    def _import_pkg(self, module: str):
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        return __import__(module)  # noqa: PLC0415

    def test_parse_free_run(self) -> None:
        history = self._import_pkg("results_history")
        parsed = history.parse_free_run("614/634 (0.9685)")
        self.assertEqual(parsed, (614, 634))

    def test_build_brief_finds_session_best(self) -> None:
        brief_mod = self._import_pkg("session_brief")

        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            config_dir = repo / "configs"
            config_dir.mkdir()
            cfg = config_dir / "iter33.json"
            cfg.write_text(
                json.dumps(
                    {
                        "run": {
                            "init_model_path": "models_wsl/ar/tide_overfit_iter/iter17/ar_onset_model.keras",
                            "lambda_incremental_consistency": 0.15,
                        },
                    },
                ),
                encoding="utf-8",
            )
            results = repo / "logs" / "ar_tide_iter" / "results.jsonl"
            results.parent.mkdir(parents=True)
            row = {
                "id": "iter33",
                "timestamp": "t1",
                "notes": "good",
                "model_path": "models_wsl/ar/tide_overfit_iter/iter33/ar_onset_model.keras",
                "config": str(cfg),
                "train_exit": 0,
                "free_run": "614/634 (0.9685)",
                "passed": False,
            }
            results.write_text(json.dumps(row) + "\n", encoding="utf-8")

            brief = brief_mod.build_brief(repo=repo, results_path=results)
            self.assertEqual(brief["session_best"]["id"], "iter33")
            self.assertEqual(brief["session_best"]["free_run_matched"], 614)
            self.assertEqual(brief["suggested_next_id"], "iter34")

    def test_load_results_reads_jsonl(self) -> None:
        history_mod = self._import_pkg("results_history")
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = pathlib.Path(tmp) / "configs"
            config_dir.mkdir()
            cfg = config_dir / "iter33.json"
            cfg.write_text(
                json.dumps(
                    {
                        "run": {
                            "lambda_incremental_consistency": 0.15,
                            "init_model_path": "models_wsl/ar/tide_overfit_iter/iter17/ar_onset_model.keras",
                        },
                    },
                ),
                encoding="utf-8",
            )
            results = pathlib.Path(tmp) / "results.jsonl"
            row = {
                "id": "iter33",
                "timestamp": "t",
                "notes": "n",
                "model_path": "models_wsl/ar/tide_overfit_iter/iter33/ar_onset_model.keras",
                "config": str(cfg),
                "train_exit": 0,
                "free_run": "614/634 (0.9685)",
                "passed": False,
            }
            results.write_text(json.dumps(row) + "\n", encoding="utf-8")
            records = history_mod.load_results(results, repo=pathlib.Path(tmp))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].matched, 614)


if __name__ == "__main__":
    unittest.main()
