"""Tests for scripts/eval_agent_setup.py."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_agent_setup as eval_mod  # noqa: E402


class LiveSuiteAndCatalogTest(unittest.TestCase):
    def test_empty_local_suites_pass_validate_unless_coverage_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            rules_dir = root / ".agents" / "rules"
            rule_evals = rules_dir / "evals"
            skills_dir = root / ".agents" / "skills"
            (skills_dir / "demo").mkdir(parents=True)
            rules_dir.mkdir(parents=True)
            (skills_dir / "demo" / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Demo skill\n---\n",
                encoding="utf-8",
            )
            paths = eval_mod.discover_suites(
                skills_dir=skills_dir, rule_evals_dir=rule_evals
            )
            self.assertEqual(paths, [])
            self.assertEqual(
                eval_mod.validate_local_suites(
                    paths,
                    rules_dir=rules_dir,
                    skills_dir=skills_dir,
                    rule_evals_dir=rule_evals,
                    check_all_coverage=False,
                ),
                [],
            )
            cov_errors = eval_mod.validate_local_suites(
                paths,
                rules_dir=rules_dir,
                skills_dir=skills_dir,
                rule_evals_dir=rule_evals,
                check_all_coverage=True,
            )
            self.assertTrue(
                any("Missing local skill eval suite" in err for err in cov_errors)
            )

    def test_single_skill_discovery_and_validation_inside_skill_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            rules_dir = root / ".agents" / "rules"
            rule_evals = rules_dir / "evals"
            skills_dir = root / ".agents" / "skills"
            (skills_dir / "demo" / "evals").mkdir(parents=True)
            rules_dir.mkdir(parents=True)
            (skills_dir / "demo" / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Demo skill\n---\n",
                encoding="utf-8",
            )
            suite_file = skills_dir / "demo" / "evals" / "suite.json"
            suite_file.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "owner": "demo",
                        "owner_type": "skill",
                        "trigger_cases": [
                            {
                                "id": "t1",
                                "prompt": "run demo",
                                "expected_skills": ["demo"],
                            },
                            {
                                "id": "t2",
                                "prompt": "unrelated",
                                "expected_skills": [],
                                "forbidden_skills": ["demo"],
                            },
                        ],
                        "behavior_cases": [],
                    }
                ),
                encoding="utf-8",
            )
            paths = eval_mod.discover_suites(
                skill="demo", skills_dir=skills_dir, rule_evals_dir=rule_evals
            )
            self.assertEqual(paths, [suite_file])
            errors = eval_mod.validate_local_suites(
                paths,
                rules_dir=rules_dir,
                skills_dir=skills_dir,
                rule_evals_dir=rule_evals,
                check_all_coverage=False,
            )
            self.assertEqual(errors, [])

    def test_load_live_catalog_finds_model_decision_rules_and_skills(self) -> None:
        cat = eval_mod.load_live_catalog()
        rule_ids = {r["id"] for r in cat["rules"]}
        skill_ids = {s["id"] for s in cat["skills"]}
        self.assertEqual(rule_ids, {"git-and-ci.md", "python-env.md"})
        self.assertIn("onset-evaluation", skill_ids)
        self.assertIn("autoresearch", skill_ids)
        autoresearch_desc = next(
            s["description"] for s in cat["skills"] if s["id"] == "autoresearch"
        )
        self.assertIn("One-prompt agent autoresearch loop", autoresearch_desc)


class ValidateLocalSuitesErrorsTest(unittest.TestCase):
    def test_flags_missing_suite_owner_mismatch_missing_neg_and_duplicates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            rules_dir = root / ".agents" / "rules"
            rule_evals = rules_dir / "evals"
            skills_dir = root / ".agents" / "skills"
            rule_evals.mkdir(parents=True)
            (rules_dir / "git.md").write_text(
                "---\ndescription: Git\ntrigger: model_decision\n---\n",
                encoding="utf-8",
            )
            (skills_dir / "alpha" / "evals").mkdir(parents=True)
            (skills_dir / "alpha" / "SKILL.md").write_text(
                "---\nname: alpha\ndescription: Alpha skill\n---\n",
                encoding="utf-8",
            )
            (skills_dir / "beta").mkdir(parents=True)
            (skills_dir / "beta" / "SKILL.md").write_text(
                "---\nname: beta\ndescription: Beta skill\n---\n",
                encoding="utf-8",
            )

            alpha_suite = {
                "schema_version": 1,
                "owner": "wrong-owner",
                "owner_type": "skill",
                "trigger_cases": [
                    {
                        "id": "dup-1",
                        "prompt": "do alpha",
                        "expected_skills": ["ghost"],
                        "forbidden_skills": [],
                    },
                    {
                        "id": "dup-1",
                        "prompt": "again",
                        "expected_skills": ["alpha"],
                        "forbidden_skills": ["alpha"],
                    },
                ],
                "behavior_cases": [
                    {
                        "id": "beh-bad",
                        "target": "no-such",
                        "prompt": "hi",
                        "must_contain_regex": ["[unclosed"],
                    }
                ],
            }
            alpha_p = skills_dir / "alpha" / "evals" / "suite.json"
            alpha_p.write_text(json.dumps(alpha_suite), encoding="utf-8")

            paths = eval_mod.discover_suites(
                skills_dir=skills_dir, rule_evals_dir=rule_evals
            )
            errors = eval_mod.validate_local_suites(
                paths,
                rules_dir=rules_dir,
                skills_dir=skills_dir,
                rule_evals_dir=rule_evals,
                check_all_coverage=True,
            )
            joined = "\n".join(errors)
            self.assertIn("Missing local skill eval suite", joined)
            self.assertIn("Missing local model_decision rule eval suite", joined)
            self.assertIn("unknown owner skill 'wrong-owner'", joined)
            self.assertIn("Duplicate case id 'dup-1'", joined)
            self.assertIn("unknown expected_skills entry 'ghost'", joined)
            self.assertIn("overlap", joined)
            self.assertIn("invalid regex", joined)


class ScoreAndHillClimbDiffTest(unittest.TestCase):
    def _sample_suite(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "owner": "onset-evaluation",
            "owner_type": "skill",
            "trigger_cases": [
                {
                    "id": "t-pos",
                    "prompt": "eval ckpt",
                    "expected_rules": [],
                    "optional_rules": ["python-env.md"],
                    "forbidden_rules": [],
                    "expected_skills": ["onset-evaluation"],
                    "optional_skills": [],
                    "forbidden_skills": [],
                },
                {
                    "id": "t-neg",
                    "prompt": "explain math",
                    "expected_rules": [],
                    "optional_rules": [],
                    "forbidden_rules": ["git-and-ci.md"],
                    "expected_skills": [],
                    "optional_skills": [],
                    "forbidden_skills": ["onset-evaluation"],
                },
            ],
            "behavior_cases": [
                {
                    "id": "b-1",
                    "target": "onset-evaluation",
                    "prompt": "check raw audio",
                    "must_contain_any": [["eval_raw_audio_onset.py"]],
                    "must_contain_regex": ["oracle"],
                    "must_not_contain_regex": ["M-slot48"],
                    "rubric": ["Compares oracle vs Beat This!"],
                }
            ],
        }

    def test_score_and_diff_detects_fixed_and_regressed(self) -> None:
        suite = self._sample_suite()
        base_preds = {
            "schema_version": 1,
            "tag": "base",
            "trigger_predictions": {
                "t-pos": {"rules": ["python-env.md"], "skills": []},
                "t-neg": {"rules": [], "skills": []},
            },
            "behavior_predictions": {
                "b-1": {
                    "response": "Use eval_raw_audio_onset.py with oracle.",
                    "commands": [],
                    "rubric_results": [
                        {
                            "criterion": "Compares oracle vs Beat This!",
                            "passed": True,
                            "reason": "ok",
                        }
                    ],
                }
            },
        }
        cand_preds = copy.deepcopy(base_preds)
        cand_preds["tag"] = "cand"
        cand_preds["trigger_predictions"]["t-pos"] = {
            "rules": ["python-env.md"],
            "skills": ["onset-evaluation"],
        }
        cand_preds["behavior_predictions"]["b-1"]["response"] += " Also M-slot48."

        base_rep = eval_mod.score_predictions(suite, base_preds)
        cand_rep = eval_mod.score_predictions(suite, cand_preds)
        self.assertEqual(base_rep["triggers"]["passed"], 1)
        self.assertEqual(cand_rep["triggers"]["passed"], 2)
        self.assertEqual(cand_rep["triggers"]["f1"], 1.0)
        self.assertTrue(base_rep["behavior"]["cases"][0]["passed"])
        self.assertFalse(cand_rep["behavior"]["cases"][0]["passed"])

        diff = eval_mod.diff_reports(base_rep, cand_rep)
        self.assertEqual(diff["fixed"], ["triggers:t-pos"])
        self.assertEqual(diff["regressed"], ["behavior:b-1"])
        self.assertEqual(diff["trigger_f1_delta_pp"], 100.0)

    def test_commands_and_code_not_contain_regex_ignore_prose(self) -> None:
        suite = {
            "schema_version": 1,
            "behavior_cases": [
                {
                    "id": "b-cmd",
                    "target": "git-and-ci.md",
                    "prompt": "push",
                    "commands_not_contain_regex": ["--no-verify"],
                },
                {
                    "id": "b-code",
                    "target": "python-style.md",
                    "prompt": "pathlib",
                    "code_not_contain_regex": ["from pathlib import"],
                },
            ],
        }
        clean_preds = {
            "behavior_predictions": {
                "b-cmd": {
                    "response": "Never pass --no-verify when pushing.",
                    "commands": ["git push origin master"],
                },
                "b-code": {
                    "response": "Never use `from pathlib import Path`.\n```python\nimport pathlib\n```",
                    "commands": ["ruff check ."],
                },
            }
        }
        clean_rep = eval_mod.score_predictions(suite, clean_preds)
        self.assertEqual(clean_rep["behavior"]["passed"], 2)

        bad_preds = {
            "behavior_predictions": {
                "b-cmd": {
                    "response": "Pushing now.",
                    "commands": ["git push --no-verify"],
                },
                "b-code": {
                    "response": "Here:\n```python\nfrom pathlib import Path\n```",
                    "commands": [],
                },
            }
        }
        bad_rep = eval_mod.score_predictions(suite, bad_preds)
        self.assertEqual(bad_rep["behavior"]["passed"], 0)

    def test_cli_score_exits_one_on_regression_and_zero_when_clean(self) -> None:
        suite = self._sample_suite()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            suite_p = root / "suite.json"
            base_p = root / "base.json"
            cand_p = root / "cand.json"
            out_p = root / "report.json"

            suite_p.write_text(json.dumps(suite), encoding="utf-8")
            good = {
                "schema_version": 1,
                "tag": "v1",
                "trigger_predictions": {
                    "t-pos": {"rules": [], "skills": ["onset-evaluation"]},
                    "t-neg": {"rules": [], "skills": []},
                },
                "behavior_predictions": {
                    "b-1": {
                        "response": "Run eval_raw_audio_onset.py against oracle.",
                        "commands": [],
                        "rubric_results": [
                            {
                                "criterion": "Compares oracle vs Beat This!",
                                "passed": True,
                                "reason": "ok",
                            }
                        ],
                    }
                },
            }
            bad = copy.deepcopy(good)
            bad["trigger_predictions"]["t-neg"]["skills"] = ["onset-evaluation"]
            base_p.write_text(json.dumps(bad), encoding="utf-8")
            cand_p.write_text(json.dumps(good), encoding="utf-8")

            with (
                mock.patch.object(eval_mod, "validate_local_suites", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                rc_fix = eval_mod.main(
                    [
                        "score",
                        "--suite",
                        str(suite_p),
                        "--predictions",
                        str(cand_p),
                        "--baseline",
                        str(base_p),
                        "--out",
                        str(out_p),
                        "--strict",
                    ]
                )
            self.assertEqual(rc_fix, 0)
            self.assertIn("FIXED (FAIL → PASS): ['triggers:t-neg']", out.getvalue())
            self.assertTrue(out_p.is_file())

            with (
                mock.patch.object(eval_mod, "validate_local_suites", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc_reg = eval_mod.main(
                    [
                        "score",
                        "--suite",
                        str(suite_p),
                        "--predictions",
                        str(base_p),
                        "--baseline",
                        str(cand_p),
                    ]
                )
            self.assertEqual(rc_reg, 1)


if __name__ == "__main__":
    unittest.main()
