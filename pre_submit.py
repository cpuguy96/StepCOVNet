"""Run the same checks as .github/workflows/pre-submit.yml locally."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


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
    """Execute pre-submit validation steps."""
    parser = argparse.ArgumentParser(
        description="Mirror GitHub Actions Pre-Submit Checks locally."
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip install -e .[dev]",
    )
    parser.add_argument("--skip-ruff", action="store_true", help="Skip ruff check .")
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
    args = parser.parse_args(argv)

    root = _repo_root()
    os.chdir(root)
    python = _default_python(root)

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

    if not args.skip_tests:
        _run_step(
            "Unit tests (pytest tests/ --cov-report=xml)",
            [str(python), "-m", "pytest", "tests/", "--cov-report=xml"],
            cwd=root,
        )

    if not args.skip_nbmake:
        _run_step(
            "Notebook tests (pytest --nbmake notebooks)",
            [str(python), "-m", "pytest", "--nbmake", "notebooks"],
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
