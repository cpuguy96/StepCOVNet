#!/usr/bin/env python3
"""Read-only audit of agent-brain artifacts (rules, skills, catalog docs).

Usage (repo root, project venv):
    python scripts/audit_agent_brain.py

Prints disk inventory and drift vs tracked indexes. Does not write files — the
agent updates docs/agents/agent-brain.md and indexes during brain refresh.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
RULES_DIR = REPO / ".cursor" / "rules"
SKILLS_DIR = REPO / ".cursor" / "skills"
CATALOG_PATH = REPO / "docs" / "agents" / "agent-brain.md"
AGENTS_MD = REPO / "AGENTS.md"
SKILLS_README = SKILLS_DIR / "README.md"

FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---", re.DOTALL)
SKILL_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")


def _parse_frontmatter(text: str) -> dict[str, str]:
    text = text.lstrip("\ufeff")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}
    block = match.group(1)
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_rules() -> list[dict[str, str]]:
    """Load rule metadata from ``.cursor/rules/*.mdc``."""
    rules: list[dict[str, str]] = []
    for path in sorted(RULES_DIR.glob("*.mdc")):
        meta = _parse_frontmatter(path.read_text(encoding="utf-8"))
        rules.append(
            {
                "file": path.name,
                "always_apply": meta.get("alwaysApply", "false"),
                "globs": meta.get("globs", ""),
                "description": meta.get("description", ""),
            }
        )
    return rules


def load_skills() -> list[str]:
    """List skill directory names that contain ``SKILL.md``."""
    return sorted(
        p.parent.name
        for p in SKILLS_DIR.glob("*/SKILL.md")
        if p.parent.name != "README"
    )


def _stale_rule_names(text: str, rule_files: set[str]) -> list[str]:
    hits: list[str] = []
    for name in re.findall(r"[\w-]+\.mdc", text):
        if name not in rule_files:
            hits.append(name)
    return sorted(set(hits))


def _skills_in_readme(text: str) -> set[str]:
    found: set[str] = set()
    for _label, href in SKILL_LINK_RE.findall(text):
        if "/SKILL.md" in href or href.endswith("SKILL.md"):
            part = href.split("/")[-2] if "/" in href else href.replace(".md", "")
            if part and part != "SKILL":
                found.add(part)
    return found


def format_disk_inventory(rules: list[dict[str, str]], skills: list[str]) -> str:
    """Return a read-only markdown summary of rules and skills on disk."""
    lines = ["## Disk inventory", ""]
    always = [r for r in rules if r["always_apply"].lower() == "true"]
    scoped = [r for r in rules if r["always_apply"].lower() != "true"]

    lines.append(f"Always-apply ({len(always)}):")
    for r in always:
        lines.append(f"  - {r['file']}: {r['description']}")

    lines.append(f"\nScoped ({len(scoped)}):")
    for r in scoped:
        globs = r["globs"] or "(no globs)"
        lines.append(f"  - {r['file']} [{globs}]: {r['description']}")

    lines.append(f"\nSkills ({len(skills)}):")
    for name in skills:
        lines.append(f"  - {name}")

    lines.append("")
    return "\n".join(lines)


def audit() -> int:
    """Print inventory and drift; return 0 if OK, 1 if drift found."""
    rules = load_rules()
    skills = load_skills()
    rule_files = {r["file"] for r in rules}
    errors: list[str] = []

    readme_skills = _skills_in_readme(SKILLS_README.read_text(encoding="utf-8"))
    missing_readme = sorted(set(skills) - readme_skills)
    if missing_readme:
        errors.append(f"Skills missing from skills README: {', '.join(missing_readme)}")

    extra_readme = sorted(readme_skills - set(skills) - {"README"})
    if extra_readme:
        errors.append(f"skills README lists missing dirs: {', '.join(extra_readme)}")

    for path, label in (
        (AGENTS_MD, "AGENTS.md"),
        (SKILLS_README, "skills README"),
        (CATALOG_PATH, "agent-brain.md"),
    ):
        if not path.is_file():
            continue
        stale = _stale_rule_names(path.read_text(encoding="utf-8"), rule_files)
        if stale:
            errors.append(f"{label} references deleted rules: {', '.join(stale)}")

    always_count = sum(1 for r in rules if r["always_apply"].lower() == "true")
    print(format_disk_inventory(rules, skills))
    print(
        f"Summary: {len(rules)} rules "
        f"({always_count} always-apply, {len(rules) - always_count} scoped), "
        f"{len(skills)} skills"
    )

    if errors:
        print("\nDRIFT:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\nOK - indexes match disk.")
    return 0


def main() -> None:
    raise SystemExit(audit())


if __name__ == "__main__":
    main()
