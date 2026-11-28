"""Run the same checks as .github/workflows/pre-submit.yml locally."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def find_code_changes_without_tests(changed_paths: Sequence[str]) -> list[str]:
    """Return ``src/`` or ``scripts/`` Python paths when no ``tests/`` file changed.

    Args:
        changed_paths: Repo-relative modified or untracked paths.

    Returns:
        Sorted list of changed Python files under ``src/`` or ``scripts/`` when
        no ``.py`` file under ``tests/`` was changed alongside them; otherwise
        an empty list.
    """
    normalized = [path.replace("\\", "/").lstrip("./") for path in changed_paths]
    code_paths = sorted(
        {
            path
            for path in normalized
            if path.endswith(".py") and path.startswith(("src/", "scripts/"))
        }
    )
    if not code_paths:
        return []
    has_test_change = any(
        path.endswith(".py") and path.startswith("tests/") for path in normalized
    )
    return [] if has_test_change else code_paths


def git_changed_paths(root: Path) -> list[str]:
    """Collect modified, added, renamed, and untracked paths via ``git status``.

    Args:
        root: Repository root directory.

    Returns:
        List of repo-relative paths excluding pure deletions (or empty list if
        ``git`` is unavailable or exits non-zero).
    """
    completed = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    paths: list[str] = []
    for raw_line in completed.stdout.splitlines():
        if len(raw_line) < 4:
            continue
        status = raw_line[:2]
        if status.strip() == "D":
            continue
        path_part = raw_line[3:].strip()
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1].strip()
        path_part = path_part.strip('"')
        if path_part:
            paths.append(path_part)
    return paths


def _default_python(root: Path) -> Path:
    candidates = (
        root / "venv" / "Scripts" / "python.exe",
        root / "venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def _run_step(label: str, command: list[str], *, cwd: Path) -> None:
    print(f"=== {label} ===", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(argv: list[str] | None = None) -> None:
    """Execute pre-submit validation steps.

    Args:
        argv: Optional CLI argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: If the test-diff gate or any enabled check fails.
    """
    parser = argparse.ArgumentParser(
        description="Mirror GitHub Actions Pre-Submit Checks locally."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Static checks only (ruff check, ruff format --check, pyright).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install -e .[dev]",
    )
    parser.add_argument(
        "--skip-ruff",
        action="store_true",
        help="Skip ruff check . and ruff format --check .",
    )
    parser.add_argument(
        "--skip-pyright",
        action="store_true",
        help="Skip pyright type check",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest tests/ --cov-report=xml",
    )
    parser.add_argument(
        "--skip-nbmake",
        action="store_true",
        help="Skip pytest --nbmake notebooks",
    )
    parser.add_argument(
        "--codacy",
        action="store_true",
        help="Upload coverage.xml to Codacy (needs CODACY_PROJECT_TOKEN)",
    )
    parser.add_argument(
        "--skip-test-diff-check",
        action="store_true",
        help="Skip checking that src/ or scripts/ Python changes include tests/ changes.",
    )
    args = parser.parse_args(argv)
    if args.fast:
        args.skip_install = True
        args.skip_tests = True
        args.skip_nbmake = True

    root = _repo_root()
    os.chdir(root)
    python = _default_python(root)

    if not args.skip_test_diff_check:
        uncovered = find_code_changes_without_tests(git_changed_paths(root))
        if uncovered:
            listed = ", ".join(uncovered)
            print(
                "Error: Python code changes under src/ or scripts/ have no "
                f"accompanying tests/ Python changes: {listed}\n"
                "Add or update unit tests in tests/ (or pass "
                "--skip-test-diff-check for pure formatting/docstring edits).",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)

    if not args.skip_install:
        _run_step(
            "Install dependencies",
            [str(python), "-m", "pip", "install", "-e", ".[dev]"],
            cwd=root,
        )

    if not args.skip_ruff:
        _run_step(
            "Ruff check (ruff check .)",
            [str(python), "-m", "ruff", "check", "."],
            cwd=root,
        )
        _run_step(
            "Ruff format check (ruff format --check .)",
            [str(python), "-m", "ruff", "format", "--check", "."],
            cwd=root,
        )

    if not args.skip_pyright:
        _run_step(
            "Pyright type check",
            [str(python), "-m", "pyright", "--pythonpath", str(python)],
            cwd=root,
        )

    if not args.skip_tests:
        _run_step(
            "Unit tests (pytest tests/ --cov-report=term --cov-report=xml)",
            [
                str(python),
                "-m",
                "pytest",
                "tests/",
                "--cov-report=term",
                "--cov-report=xml",
            ],
            cwd=root,
        )

    if not args.skip_nbmake:
        _run_step(
            "Notebook tests (pytest --nbmake notebooks --no-cov)",
            [str(python), "-m", "pytest", "--nbmake", "notebooks", "--no-cov"],
            cwd=root,
        )

    if args.codacy:
        token = os.environ.get("CODACY_PROJECT_TOKEN")
        if not token:
            print(
                "CODACY_PROJECT_TOKEN is not set; skipping Codacy upload.",
                file=sys.stderr,
            )
        else:
            _run_step(
                "Codacy coverage upload",
                [
                    "bash",
                    "-lc",
                    "bash <(curl -Ls https://coverage.codacy.com/get.sh) report -r coverage.xml",
                ],
                cwd=root,
            )

    print("Pre-submit checks passed.", flush=True)


if __name__ == "__main__":
    main()
