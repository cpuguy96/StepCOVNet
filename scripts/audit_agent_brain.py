#!/usr/bin/env python3
"""Read-only audit of StepCOVNet agent customization files.

Usage (repo root, project venv):
    python scripts/audit_agent_brain.py

Verifies rule and skill frontmatter, relative markdown link integrity, router
and always-on context budgets, and legacy/path portability hygiene.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
RULES_DIR = REPO / ".agents" / "rules"
SKILLS_DIR = REPO / ".agents" / "skills"
AGENTS_MD = REPO / "AGENTS.md"
PROJECT_LAYOUT = REPO / "docs" / "agents" / "project-layout.md"

MAX_AGENTS_MD_LINES = 65
MAX_ALWAYS_ON_RULES = 2

FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---", re.DOTALL)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
LEGACY_RULE_RE = re.compile(r"[\w-]+\.mdc\b")
CURSOR_DIR_RE = re.compile(r"\.cursor/")
ABS_USER_PATH_RE = re.compile(r"(?:[A-Za-z]:\\Users\\|/mnt/[a-z]/Users/)")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract top YAML frontmatter key-value pairs from markdown text.

    Args:
        text: Raw markdown file contents.

    Returns:
        Mapping of frontmatter keys to parsed string values.
    """
    text = text.lstrip("\ufeff")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}

    lines = match.group(1).splitlines()
    out: dict[str, str] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if ":" not in line or line.startswith((" ", "\t")):
            idx += 1
            continue
        key, _, raw_val = line.partition(":")
        key = key.strip()
        val = raw_val.strip()
        if val in {">", ">-", "|", "|-"}:
            parts: list[str] = []
            idx += 1
            while idx < len(lines) and (
                lines[idx].startswith((" ", "\t")) or not lines[idx].strip()
            ):
                stripped = lines[idx].strip()
                if stripped:
                    parts.append(stripped)
                idx += 1
            out[key] = " ".join(parts)
            continue
        out[key] = val.strip('"').strip("'")
        idx += 1
    return out


def load_rules() -> list[dict[str, str]]:
    """Load rule metadata from ``.agents/rules/*.md``.

    Returns:
        Sorted rule metadata dicts with ``file``, ``trigger``, ``always_apply``,
        ``globs``, and ``description`` keys.
    """
    rules: list[dict[str, str]] = []
    if not RULES_DIR.is_dir():
        return rules
    for path in sorted(RULES_DIR.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        meta = _parse_frontmatter(path.read_text(encoding="utf-8"))
        trigger = meta.get("trigger", "").lower()
        always_apply = (
            "true"
            if trigger == "always_on" or meta.get("alwaysApply", "").lower() == "true"
            else "false"
        )
        rules.append(
            {
                "file": path.name,
                "trigger": trigger,
                "always_apply": always_apply,
                "globs": meta.get("globs", ""),
                "description": meta.get("description", ""),
            }
        )
    return rules


def load_skills() -> list[dict[str, str]]:
    """Load skill metadata from ``.agents/skills/*/SKILL.md``.

    Returns:
        Sorted skill metadata dicts with ``dir``, ``name``, and ``description``.
    """
    skills: list[dict[str, str]] = []
    if not SKILLS_DIR.is_dir():
        return skills
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        meta = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        skills.append(
            {
                "dir": skill_md.parent.name,
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
            }
        )
    return skills


def _broken_relative_links(
    source_path: pathlib.Path,
    text: str,
    repo_root: pathlib.Path,
) -> list[str]:
    """Find broken relative markdown links in ``text``.

    Args:
        source_path: Markdown file path containing ``text``.
        text: Raw markdown file contents.
        repo_root: Repository root boundary for resolved targets.

    Returns:
        Sorted list of broken link ``href`` strings.
    """
    stripped = FENCED_CODE_RE.sub("", text)
    stripped = INLINE_CODE_RE.sub("", stripped)
    repo_resolved = repo_root.resolve()
    broken: set[str] = set()
    for raw_href in MARKDOWN_LINK_RE.findall(stripped):
        href = raw_href.strip()
        if not href or href.startswith(
            ("http://", "https://", "mailto:", "file://", "#", "<")
        ):
            continue
        target_part = href.split("#", 1)[0].strip()
        if not target_part:
            continue
        resolved = (source_path.parent / target_part).resolve()
        if not resolved.is_relative_to(repo_resolved) or not resolved.exists():
            broken.add(href)
    return sorted(broken)


def _hygiene_violations(text: str) -> list[str]:
    """Collect legacy Cursor references and machine-specific paths in ``text``.

    Args:
        text: Markdown contents of an active agent configuration file.

    Returns:
        Sorted list of matched legacy or non-portable substrings.
    """
    hits: set[str] = set()
    hits.update(LEGACY_RULE_RE.findall(text))
    hits.update(CURSOR_DIR_RE.findall(text))
    hits.update(ABS_USER_PATH_RE.findall(text))
    return sorted(hits)


def _active_markdown_files() -> list[pathlib.Path]:
    """Collect active agent markdown files subject to link and hygiene checks.

    Returns:
        Existing markdown file paths across ``AGENTS.md``, ``.agents/``, and
        ``docs/agents/project-layout.md``.
    """
    files: list[pathlib.Path] = []
    if AGENTS_MD.is_file():
        files.append(AGENTS_MD)
    if RULES_DIR.is_dir():
        files.extend(sorted(RULES_DIR.glob("*.md")))
    if SKILLS_DIR.is_dir():
        files.extend(sorted(SKILLS_DIR.rglob("*.md")))
    if PROJECT_LAYOUT.is_file():
        files.append(PROJECT_LAYOUT)
    return files


def _validate_rules(rules: list[dict[str, str]]) -> list[str]:
    """Validate rule frontmatter fields and always-on budget.

    Args:
        rules: Rule metadata dicts returned by ``load_rules()``.

    Returns:
        Human-readable validation error strings.
    """
    errors: list[str] = []
    always_count = 0
    for rule in rules:
        name = rule["file"]
        if not rule["description"]:
            errors.append(f".agents/rules/{name}: missing frontmatter description")
        trigger = rule["trigger"]
        if trigger not in {"always_on", "glob"}:
            errors.append(
                f".agents/rules/{name}: trigger must be 'always_on' or 'glob' (got {trigger!r})"
            )
        if trigger == "glob" and not rule["globs"]:
            errors.append(f".agents/rules/{name}: trigger 'glob' requires globs")
        if rule["always_apply"] == "true":
            always_count += 1
    if always_count > MAX_ALWAYS_ON_RULES:
        errors.append(
            f"Too many always_on rules ({always_count} > {MAX_ALWAYS_ON_RULES}); scope or merge them"
        )
    return errors


def _validate_skills(skills: list[dict[str, str]]) -> list[str]:
    """Validate skill directories and ``SKILL.md`` frontmatter fields.

    Args:
        skills: Skill metadata dicts returned by ``load_skills()``.

    Returns:
        Human-readable validation error strings.
    """
    errors: list[str] = []
    if SKILLS_DIR.is_dir():
        for child in sorted(SKILLS_DIR.iterdir()):
            if child.is_dir() and not (child / "SKILL.md").is_file():
                errors.append(f".agents/skills/{child.name}: missing SKILL.md")
    for skill in skills:
        skill_dir = skill["dir"]
        if skill["name"] != skill_dir:
            errors.append(
                f".agents/skills/{skill_dir}/SKILL.md: name {skill['name']!r} != folder {skill_dir!r}"
            )
        if not skill["description"]:
            errors.append(
                f".agents/skills/{skill_dir}/SKILL.md: missing frontmatter description"
            )
    return errors


def format_disk_inventory(
    rules: list[dict[str, str]],
    skills: list[dict[str, str]] | list[str],
) -> str:
    """Format a read-only markdown summary of rules and skills on disk.

    Args:
        rules: Rule metadata dicts loaded from disk.
        skills: Skill metadata dicts or sorted skill names.

    Returns:
        Formatted markdown inventory string.
    """
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
    for item in skills:
        label = item["dir"] if isinstance(item, dict) else item
        lines.append(f"  - {label}")

    lines.append("")
    return "\n".join(lines)


def audit() -> int:
    """Print inventory and validation errors for the agent setup.

    Returns:
        Process exit code (``0`` if clean, ``1`` if violations are detected).
    """
    rules = load_rules()
    skills = load_skills()
    errors: list[str] = []
    errors.extend(_validate_rules(rules))
    errors.extend(_validate_skills(skills))

    if AGENTS_MD.is_file():
        line_count = len(AGENTS_MD.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_AGENTS_MD_LINES:
            errors.append(
                f"AGENTS.md has {line_count} lines (limit {MAX_AGENTS_MD_LINES}); keep it a thin router"
            )

    repo_resolved = REPO.resolve()
    for path in _active_markdown_files():
        text = path.read_text(encoding="utf-8")
        try:
            rel = path.resolve().relative_to(repo_resolved).as_posix()
        except ValueError:
            rel = path.as_posix()
        for href in _broken_relative_links(path, text, REPO):
            errors.append(f"{rel}: broken relative link ({href})")
        for hit in _hygiene_violations(text):
            errors.append(f"{rel}: disallowed legacy/path token ({hit})")

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

    print("\nOK - agent setup clean.")
    return 0


def main() -> None:
    raise SystemExit(audit())


if __name__ == "__main__":
    main()
