from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_agent_setup  # noqa: E402
import run_agent_eval  # noqa: E402


class RunAgentEvalTest(unittest.TestCase):
    def test_extract_json_object_handles_raw_fenced_and_invalid(self) -> None:
        self.assertEqual(run_agent_eval.extract_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(
            run_agent_eval.extract_json_object(
                'Here:\n```json\n{"rules": ["x.md"]}\n```'
            ),
            {"rules": ["x.md"]},
        )
        with self.assertRaises(ValueError):
            run_agent_eval.extract_json_object("[1, 2, 3]")
        with self.assertRaises(ValueError):
            run_agent_eval.extract_json_object("no json here")

    def test_prompt_builders_and_bundle_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = pathlib.Path(tmpdir) / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "owner": "git-and-ci.md",
                        "owner_type": "rule",
                        "trigger_cases": [
                            {"id": "t1", "prompt": "Commit files"},
                            {"id": "t2", "prompt": "Other prompt"},
                        ],
                        "behavior_cases": [
                            {
                                "id": "b1",
                                "target": "git-and-ci.md",
                                "prompt": "Push fix",
                                "rubric": ["Add tests first."],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            catalog = {
                "rules": [{"id": "git-and-ci.md", "description": "Git rules"}],
                "skills": [{"id": "demo", "description": "Demo skill"}],
            }
            catalog_md = eval_agent_setup.format_catalog_markdown(catalog)
            bundle = run_agent_eval.build_prompt_bundle(
                [suite_path],
                catalog,
                catalog_md,
                mode="all",
                case_ids={"t1", "b1"},
            )
            self.assertEqual([item["id"] for item in bundle["trigger_prompts"]], ["t1"])
            self.assertIn("Commit files", bundle["trigger_prompts"][0]["prompt"])
            self.assertEqual(
                [item["id"] for item in bundle["behavior_prompts"]], ["b1"]
            )
            grader_prompt = run_agent_eval.build_grader_prompt(
                bundle["behavior_prompts"][0]["case"],
                {"response": "Run pytest", "commands": ["pytest"]},
            )
            self.assertIn("Add tests first.", grader_prompt)
            self.assertIn("Run pytest", grader_prompt)

    def test_run_cmd_backend_passes_stdin_and_schema_env(self) -> None:
        completed = subprocess.CompletedProcess(
            args="dummy-cli",
            returncode=0,
            stdout='{"reason": "ok", "rules": [], "skills": []}',
            stderr="",
        )
        with mock.patch.object(
            run_agent_eval.subprocess,
            "run",
            autospec=True,
            return_value=completed,
        ) as mock_run:
            out = run_agent_eval.run_cmd_backend(
                "hello prompt",
                run_agent_eval.TRIGGER_SCHEMA,
                "dummy-cli --json",
                cwd=pathlib.Path(),
            )
        self.assertEqual(out, {"reason": "ok", "rules": [], "skills": []})
        call_kwargs = mock_run.call_args.kwargs
        self.assertEqual(call_kwargs["input"], "hello prompt")
        self.assertEqual(
            json.loads(call_kwargs["env"]["EVAL_SCHEMA_JSON"]),
            run_agent_eval.TRIGGER_SCHEMA,
        )

    def test_main_dump_and_cmd_execution_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            rules_dir = root / ".agents" / "rules"
            evals_dir = rules_dir / "evals"
            evals_dir.mkdir(parents=True)
            (rules_dir / "git-and-ci.md").write_text(
                "---\ndescription: Git CI\ntrigger: model_decision\n---\n",
                encoding="utf-8",
            )
            (evals_dir / "git-and-ci.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "owner": "git-and-ci.md",
                        "owner_type": "rule",
                        "trigger_cases": [
                            {
                                "id": "t1",
                                "prompt": "Commit changes",
                                "expected_rules": ["git-and-ci.md"],
                                "optional_rules": [],
                                "forbidden_rules": [],
                                "expected_skills": [],
                                "optional_skills": [],
                                "forbidden_skills": [],
                            }
                        ],
                        "behavior_cases": [
                            {
                                "id": "b1",
                                "target": "git-and-ci.md",
                                "prompt": "Can I push without tests?",
                                "must_contain_any": [["pytest"]],
                                "rubric": ["Requires tests."],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rc_dump = run_agent_eval.main(
                [
                    "--repo-root",
                    str(root),
                    "--rule",
                    "git-and-ci.md",
                    "--tag",
                    "unit_dump",
                    "--backend",
                    "dump",
                ]
            )
            self.assertEqual(rc_dump, 0)
            bundle_path = root / "logs" / "agent_evals" / "unit_dump_prompts.json"
            self.assertTrue(bundle_path.is_file())

            fake_responses = [
                {"reason": "matches git", "rules": ["git-and-ci.md"], "skills": []},
                {"response": "No, run pytest first.", "commands": ["pytest tests/"]},
                {
                    "results": [
                        {
                            "criterion": "Requires tests.",
                            "passed": True,
                            "reason": "Checked",
                        }
                    ]
                },
            ]
            with (
                mock.patch.object(
                    run_agent_eval,
                    "run_cmd_backend",
                    autospec=True,
                    side_effect=fake_responses,
                ),
                mock.patch.object(
                    run_agent_eval.eval_agent_setup,
                    "main",
                    autospec=True,
                    return_value=0,
                ) as mock_score,
            ):
                rc_cmd = run_agent_eval.main(
                    [
                        "--repo-root",
                        str(root),
                        "--rule",
                        "git-and-ci.md",
                        "--tag",
                        "unit_cmd",
                        "--backend",
                        "cmd",
                        "--cmd",
                        "fake-agent",
                    ]
                )
            self.assertEqual(rc_cmd, 0)
            mock_score.assert_called_once()
            pred_path = root / "logs" / "agent_evals" / "unit_cmd.json"
            preds = json.loads(pred_path.read_text(encoding="utf-8"))
            self.assertEqual(
                preds["trigger_predictions"]["t1"]["rules"], ["git-and-ci.md"]
            )
            self.assertEqual(
                preds["behavior_predictions"]["b1"]["rubric_results"][0]["passed"],
                True,
            )


if __name__ == "__main__":
    unittest.main()
