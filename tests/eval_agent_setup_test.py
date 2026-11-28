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
    def test_live_suite_validates_cleanly(self) -> None:
        suite = eval_mod.load_json(eval_mod.DEFAULT_SUITE)
        errors = eval_mod.validate_suite(suite)
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


class ValidateSuiteErrorsTest(unittest.TestCase):
    def test_flags_schema_duplicate_overlap_unknown_and_uncovered(self) -> None:
        catalog = {
            "rules": [
                {"id": "git.md", "description": "Git"},
                {"id": "uncovered.md", "description": "Unused"},
            ],
            "skills": [
                {"id": "alpha", "description": "Alpha"},
                {"id": "missing-cov", "description": "Uncovered skill"},
            ],
        }
        suite = {
            "schema_version": 99,
            "trigger_cases": [
                {
                    "id": "c1",
                    "prompt": "run git",
                    "expected_rules": ["git.md"],
                    "forbidden_rules": ["git.md"],
                    "expected_skills": ["alpha", "ghost"],
                },
                {
                    "id": "c1",
                    "prompt": "",
                    "expected_rules": [],
                    "expected_skills": [],
                },
            ],
            "behavior_cases": [
                {
                    "id": "b1",
                    "target": "no-such-target",
                    "prompt": "hi",
                    "must_contain_regex": ["[unclosed"],
                }
            ],
        }
        errors = eval_mod.validate_suite(suite, catalog=catalog)
        joined = "\n".join(errors)
        self.assertIn("schema_version", joined)
        self.assertIn("Duplicate case id 'c1'", joined)
        self.assertIn("unknown expected_skills entry 'ghost'", joined)
        self.assertIn("overlap", joined)
        self.assertIn("Uncovered model_decision rule", joined)
        self.assertIn("Uncovered skill", joined)
        self.assertIn("invalid regex", joined)


class ScoreAndHillClimbDiffTest(unittest.TestCase):
    def _sample_suite(self) -> dict[str, object]:
        return {
            "schema_version": 1,
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
                mock.patch.object(eval_mod, "validate_suite", return_value=[]),
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
                mock.patch.object(eval_mod, "validate_suite", return_value=[]),
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
