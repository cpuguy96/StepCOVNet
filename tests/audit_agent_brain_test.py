"""Tests for scripts/audit_agent_brain.py."""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import tempfile
import unittest
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
            "trigger: glob\n"
            "---\n"
            "# body\n"
        )
        meta = brain._parse_frontmatter(text)
        self.assertEqual(meta["globs"], "**/*.py")
        self.assertEqual(meta["trigger"], "glob")

    def test_parses_multiline_folded_description(self) -> None:
        text = (
            "---\n"
            "name: autoresearch\n"
            "description: >-\n"
            "  One-prompt research loop.\n"
            "  Stops on deadline.\n"
            "---\n"
        )
        meta = brain._parse_frontmatter(text)
        self.assertEqual(meta["name"], "autoresearch")
        self.assertEqual(
            meta["description"], "One-prompt research loop. Stops on deadline."
        )

    def test_strips_bom(self) -> None:
        text = "\ufeff---\ndescription: Workspace\ntrigger: always_on\n---\n"
        meta = brain._parse_frontmatter(text)
        self.assertEqual(meta["description"], "Workspace")
        self.assertEqual(meta["trigger"], "always_on")

    def test_returns_empty_when_missing_frontmatter(self) -> None:
        self.assertEqual(brain._parse_frontmatter("# no yaml\n"), {})


class LinkAndHygieneTest(unittest.TestCase):
    def test_broken_relative_links_ignores_code_urls_and_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            target = docs / "guide.md"
            target.write_text("# Guide\n", encoding="utf-8")
            src = root / "AGENTS.md"
            text = (
                "[ok](docs/guide.md#section)\n"
                "[url](https://example.com/missing.md)\n"
                "[anchor](#top)\n"
                "`[inline](missing_in_code.md)`\n"
                "```\n[fenced](also_ignored.md)\n```\n"
                "[bad](docs/nope.md)\n"
            )
            self.assertEqual(
                brain._broken_relative_links(src, text, root), ["docs/nope.md"]
            )

    def test_hygiene_violations_flags_mdc_cursor_and_abs_paths(self) -> None:
        win_prefix = "C:\\" + "Users\\"
        wsl_prefix = "/mnt/" + "c/Users/"
        text = f"see old.mdc, .cursor/rules, {win_prefix}alice, {wsl_prefix}bob"
        hits = brain._hygiene_violations(text)
        self.assertEqual(
            hits,
            [".cursor/", wsl_prefix, win_prefix, "old.mdc"],
        )


class FormatDiskInventoryTest(unittest.TestCase):
    def test_lists_always_and_scoped_rules_and_skills(self) -> None:
        rules = [
            {
                "file": "workspace.md",
                "trigger": "always_on",
                "always_apply": "true",
                "globs": "",
                "description": "Universal",
            },
            {
                "file": "scripts.md",
                "trigger": "glob",
                "always_apply": "false",
                "globs": "scripts/**",
                "description": "Scripts",
            },
        ]
        skills = [{"dir": "demo-skill", "name": "demo-skill", "description": "Demo"}]
        text = brain.format_disk_inventory(rules, skills)
        self.assertIn("Always-apply (1)", text)
        self.assertIn("workspace.md", text)
        self.assertIn("Scoped (1)", text)
        self.assertIn("demo-skill", text)


class AuditFixtureTest(unittest.TestCase):
    def _write_minimal_repo(self, root: pathlib.Path) -> None:
        rules = root / ".agents" / "rules"
        skills = root / ".agents" / "skills" / "demo-skill"
        docs_agents = root / "docs" / "agents"
        rules.mkdir(parents=True)
        skills.mkdir(parents=True)
        docs_agents.mkdir(parents=True)
        (rules / "workspace.md").write_text(
            "---\ndescription: Universal\ntrigger: always_on\n---\n",
            encoding="utf-8",
        )
        (skills / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: Demo workflow\n---\n# demo\n",
            encoding="utf-8",
        )
        (docs_agents / "project-layout.md").write_text("# Layout\n", encoding="utf-8")
        (root / "AGENTS.md").write_text(
            "# Router\n[demo](.agents/skills/demo-skill/SKILL.md)\n",
            encoding="utf-8",
        )

    def _patch_repo(
        self, root: pathlib.Path
    ) -> contextlib.AbstractContextManager[object]:
        return mock.patch.multiple(
            brain,
            REPO=root,
            RULES_DIR=root / ".agents" / "rules",
            SKILLS_DIR=root / ".agents" / "skills",
            AGENTS_MD=root / "AGENTS.md",
            PROJECT_LAYOUT=root / "docs" / "agents" / "project-layout.md",
        )

    def test_audit_passes_on_minimal_aligned_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            out = io.StringIO()
            with self._patch_repo(root), contextlib.redirect_stdout(out):
                code = brain.audit()
            self.assertEqual(code, 0)
            self.assertIn("demo-skill", out.getvalue())

    def test_audit_fails_on_invalid_rule_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            (root / ".agents" / "rules" / "bad.md").write_text(
                "---\ntrigger: glob\n---\n", encoding="utf-8"
            )
            with (
                self._patch_repo(root),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                code = brain.audit()
            self.assertEqual(code, 1)
            self.assertIn("missing frontmatter description", out.getvalue())
            self.assertIn("requires globs", out.getvalue())

    def test_audit_fails_on_too_many_always_on_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            rules = root / ".agents" / "rules"
            for idx in range(2):
                (rules / f"extra{idx}.md").write_text(
                    f"---\ndescription: Extra {idx}\ntrigger: always_on\n---\n",
                    encoding="utf-8",
                )
            with (
                self._patch_repo(root),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                code = brain.audit()
            self.assertEqual(code, 1)
            self.assertIn("Too many always_on rules", out.getvalue())

    def test_audit_fails_on_skill_name_mismatch_and_missing_skill_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            bad_skill = root / ".agents" / "skills" / "wrong-name"
            bad_skill.mkdir()
            (bad_skill / "SKILL.md").write_text(
                "---\nname: other-name\n---\n", encoding="utf-8"
            )
            (root / ".agents" / "skills" / "empty-dir").mkdir()
            with (
                self._patch_repo(root),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                code = brain.audit()
            self.assertEqual(code, 1)
            self.assertIn("!= folder 'wrong-name'", out.getvalue())
            self.assertIn("missing frontmatter description", out.getvalue())
            self.assertIn("empty-dir: missing SKILL.md", out.getvalue())

    def test_audit_fails_on_broken_link_hygiene_and_bloated_agents_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            self._write_minimal_repo(root)
            filler = "\n".join(
                f"line {i}" for i in range(brain.MAX_AGENTS_MD_LINES + 1)
            )
            (root / "AGENTS.md").write_text(
                f"[broken](missing.md)\n.cursor/rules\n{filler}\n",
                encoding="utf-8",
            )
            with (
                self._patch_repo(root),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                code = brain.audit()
            self.assertEqual(code, 1)
            self.assertIn("broken relative link (missing.md)", out.getvalue())
            self.assertIn("disallowed legacy/path token (.cursor/)", out.getvalue())
            self.assertIn("keep it a thin router", out.getvalue())


if __name__ == "__main__":
    unittest.main()
