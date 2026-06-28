"""Tests for scripts/audit_agent_brain.py."""

from __future__ import annotations

import io
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import audit_agent_brain as brain  # noqa: E402


class ParseFrontmatterTest(unittest.TestCase):
    def test_parses_quoted_globs(self) -> None:
        text = (
            "---\n"
            "description: Lint for Python\n"
            'globs: "**/*.py"\n'
            "alwaysApply: false\n"
            "---\n"
            "# body\n"
        )
        meta = brain._parse_frontmatter(text)
        self.assertEqual(meta["globs"], "**/*.py")
        self.assertEqual(meta["alwaysApply"], "false")

    def test_strips_bom(self) -> None:
        text = "\ufeff---\ndescription: Entry routing\nalwaysApply: true\n---\n"
        meta = brain._parse_frontmatter(text)
        self.assertEqual(meta["description"], "Entry routing")
        self.assertEqual(meta["alwaysApply"], "true")

    def test_returns_empty_when_missing_frontmatter(self) -> None:
        self.assertEqual(brain._parse_frontmatter("# no yaml\n"), {})


class StaleRuleNamesTest(unittest.TestCase):
    def test_flags_unknown_mdc_references(self) -> None:
        stale = brain._stale_rule_names(
            "see deleted-rule.mdc and kept.mdc",
            {"kept.mdc"},
        )
        self.assertEqual(stale, ["deleted-rule.mdc"])


class SkillsInReadmeTest(unittest.TestCase):
    def test_extracts_skill_directory_names(self) -> None:
        text = "| x | [foo/SKILL.md](steering-correction-promotion/SKILL.md) |"
        found = brain._skills_in_readme(text)
        self.assertEqual(found, {"steering-correction-promotion"})


class FormatDiskInventoryTest(unittest.TestCase):
    def test_lists_always_and_scoped_rules(self) -> None:
        rules = [
            {
                "file": "entry.mdc",
                "always_apply": "true",
                "globs": "",
                "description": "Route",
            },
            {
                "file": "scoped.mdc",
                "always_apply": "false",
                "globs": "scripts/**",
                "description": "Scripts",
            },
        ]
        text = brain.format_disk_inventory(rules, ["demo-skill"])
        self.assertIn("Always-apply (1)", text)
        self.assertIn("entry.mdc", text)
        self.assertIn("Scoped (1)", text)
        self.assertIn("demo-skill", text)


class AuditFixtureTest(unittest.TestCase):
    def _write_minimal_repo(self, root: pathlib.Path) -> None:
        rules = root / ".cursor" / "rules"
        skills = root / ".cursor" / "skills" / "demo-skill"
        rules.mkdir(parents=True)
        skills.mkdir(parents=True)
        (rules / "entry.mdc").write_text(
            "---\ndescription: Entry\nalwaysApply: true\n---\n",
            encoding="utf-8",
        )
        (skills / "SKILL.md").write_text("# demo\n", encoding="utf-8")
        (root / "AGENTS.md").write_text("# router\n", encoding="utf-8")
        (root / ".cursor" / "skills" / "README.md").write_text(
            "| x | [demo](demo-skill/SKILL.md) |\n",
            encoding="utf-8",
        )

    def _patch_repo(self, root: pathlib.Path) -> mock._patch_dict:
        return mock.patch.multiple(
            brain,
            REPO=root,
            RULES_DIR=root / ".cursor" / "rules",
            SKILLS_DIR=root / ".cursor" / "skills",
            CATALOG_PATH=root / "docs/agents/agent-brain.md",
            AGENTS_MD=root / "AGENTS.md",
            SKILLS_README=root / ".cursor/skills/README.md",
        )

    def test_audit_passes_on_minimal_aligned_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            with self._patch_repo(root), redirect_stdout(io.StringIO()) as out:
                code = brain.audit()
            self.assertEqual(code, 0)
            self.assertIn("demo-skill", out.getvalue())

    def test_audit_fails_when_readme_missing_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            (root / ".cursor" / "skills" / "README.md").write_text(
                "| x | [other](other-skill/SKILL.md) |\n",
                encoding="utf-8",
            )
            with self._patch_repo(root), redirect_stdout(io.StringIO()):
                code = brain.audit()
            self.assertEqual(code, 1)

    def test_audit_fails_on_stale_rule_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            (root / "AGENTS.md").write_text("old-rule.mdc\n", encoding="utf-8")
            with self._patch_repo(root), redirect_stdout(io.StringIO()):
                code = brain.audit()
            self.assertEqual(code, 1)

    def test_many_always_apply_rules_do_not_fail_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            rules = root / ".cursor" / "rules"
            for idx in range(7):
                (rules / f"rule{idx}.mdc").write_text(
                    f"---\ndescription: R{idx}\nalwaysApply: true\n---\n",
                    encoding="utf-8",
                )
            with self._patch_repo(root), redirect_stdout(io.StringIO()):
                code = brain.audit()
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
