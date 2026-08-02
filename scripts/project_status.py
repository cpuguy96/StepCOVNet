#!/usr/bin/env python3
"""Local project status for orientation ("what to do now?") checks.

Usage (repo root, project venv):
    python scripts/project_status.py
    python scripts/project_status.py --json

Prints strategic fields parsed from ``docs/research/EXPERIMENT_LOG.md`` § Current
phase plus machine checks (git, GPU lock, prerequisite paths, recent logs).
Read-only — does not write files.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime

REPO = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENT_LOG = REPO / "docs" / "research" / "EXPERIMENT_LOG.md"
LOGS_DIR = REPO / "logs"

PHASE_FIELD_RE = re.compile(
    r"^\*\*([^:]+):\*\*\s*(.+)$",
    re.MULTILINE,
)
EXP_INDEX_RE = re.compile(
    r"^\| (EXP-\d{8}-\d{2}) \|",
    re.MULTILINE,
)
LADDER_CONFIGS = (
    "configs/ar/ladder_10t_50v.json",
    "configs/ar/ladder_50t_50v.json",
    "configs/ar/ladder_200t_50v.json",
    "configs/ar/ladder_300t_50v.json",
)


def _parse_current_phase(text: str) -> dict[str, str]:
    """Extract compact routing fields from § Current phase."""
    start = text.find("## Current phase")
    if start < 0:
        return {}
    subsection = text.find("\n### ", start)
    end = text.find("\n---", start)
    if subsection >= 0 and (end < 0 or subsection < end):
        end = subsection
    section = text[start:end] if end >= 0 else text[start:]
    fields: dict[str, str] = {}
    for match in PHASE_FIELD_RE.finditer(section):
        label = match.group(1).strip()
        key = label.lower().replace(" ", "_")
        fields[key] = match.group(2).strip()
    return fields


def _git_summary() -> dict[str, str | bool]:
    """Return branch name and whether the working tree is clean."""
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty_lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "branch": branch.stdout.strip() or "?",
        "clean": len(dirty_lines) == 0,
        "dirty_count": len(dirty_lines),
    }


def _gpu_lock_summary() -> dict[str, object]:
    """Report WSL GPU lock state without importing TensorFlow."""
    lock_path = LOGS_DIR / "gpu_wsl.lock"
    if not lock_path.is_file():
        return {"held": False, "path": str(lock_path.relative_to(REPO))}
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "held": True,
            "path": str(lock_path.relative_to(REPO)),
            "job": "?",
            "error": "lock file unreadable",
        }
    return {
        "held": True,
        "path": str(lock_path.relative_to(REPO)),
        "job": payload.get("job", "?"),
        "pid": payload.get("pid", "?"),
        "started_at": payload.get("started_at", "?"),
    }


def _paths_from_ladder_configs() -> list[pathlib.Path]:
    """Collect configs and training-index manifests required before ladder runs."""
    paths: list[pathlib.Path] = []
    seen: set[str] = set()
    for rel in LADDER_CONFIGS:
        cfg_path = REPO / rel
        rel_key = rel
        if rel_key not in seen:
            seen.add(rel_key)
            paths.append(cfg_path)
        if not cfg_path.is_file():
            continue
        try:
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        dataset = payload.get("dataset") or {}
        value = dataset.get("training_index_path")
        if isinstance(value, str) and value.strip():
            p = REPO / value
            rel_key = str(p.relative_to(REPO))
            if rel_key not in seen:
                seen.add(rel_key)
                paths.append(p)
    main_index = REPO / "data" / "final_data" / "training_index.json"
    rel_key = str(main_index.relative_to(REPO))
    if rel_key not in seen:
        paths.append(main_index)
    return paths


def _check_prerequisites() -> list[dict[str, str | bool]]:
    """Mark tracked paths as present or missing."""
    rows: list[dict[str, str | bool]] = []
    for path in _paths_from_ladder_configs():
        rel = str(path.relative_to(REPO))
        exists = path.is_file() or path.is_dir()
        rows.append(
            {"path": rel, "exists": exists, "kind": "file" if path.suffix else "dir"}
        )
    return rows


def _latest_exp(text: str) -> str | None:
    """Return the newest EXP id from the experiment index table."""
    for match in EXP_INDEX_RE.finditer(text):
        return match.group(1)
    return None


def _recent_logs(limit: int = 3) -> list[dict[str, str]]:
    """Return newest log files under ``logs/`` by mtime."""
    if not LOGS_DIR.is_dir():
        return []
    files = sorted(
        (p for p in LOGS_DIR.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    rows: list[dict[str, str]] = []
    for path in files[:limit]:
        mtime = path.stat().st_mtime
        rows.append(
            {
                "path": str(path.relative_to(REPO)),
                "modified": pathlib.Path(path).stat().st_mtime,
                "modified_iso": _mtime_iso(mtime),
            }
        )
    return rows


def _mtime_iso(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")


def _manifest_summary() -> dict[str, object] | None:
    """Summarize on-disk ``training_index.json`` row counts when present."""
    index_path = REPO / "data" / "final_data" / "training_index.json"
    if not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"path": str(index_path.relative_to(REPO)), "error": "unreadable"}
    counts = payload.get("counts") or {}
    songs = counts.get("songs") or {}
    rows = counts.get("rows") or {}
    return {
        "path": str(index_path.relative_to(REPO)),
        "train_rows": rows.get("train"),
        "val_rows": rows.get("val"),
        "train_songs": songs.get("train"),
        "val_songs": songs.get("val"),
        "created_at": payload.get("created_at"),
    }


def collect_status() -> dict[str, object]:
    """Gather strategic and local status into one dict."""
    log_text = ""
    if EXPERIMENT_LOG.is_file():
        log_text = EXPERIMENT_LOG.read_text(encoding="utf-8")
    phase = _parse_current_phase(log_text)
    prereqs = _check_prerequisites()
    missing = [row["path"] for row in prereqs if not row["exists"]]
    return {
        "phase": phase,
        "git": _git_summary(),
        "gpu_lock": _gpu_lock_summary(),
        "prerequisites": prereqs,
        "missing_prerequisites": missing,
        "latest_exp": _latest_exp(log_text),
        "recent_logs": _recent_logs(),
        "manifest": _manifest_summary(),
    }


def _format_markdown(status: dict[str, object]) -> str:
    """Render status dict as markdown for agent/user consumption."""
    lines: list[str] = ["# Project status", ""]
    phase = status.get("phase") or {}
    if phase:
        lines.append("## Strategic (EXPERIMENT_LOG § Current phase)")
        for label, key in (
            ("Updated", "updated"),
            ("Primary track", "primary_track"),
            ("Next action", "next_action"),
            ("Blockers", "blockers"),
            ("Alternate track", "alternate_track"),
        ):
            value = phase.get(key)
            if value:
                lines.append(f"- **{label}:** {value}")
        defer_keys = [k for k in phase if k.startswith("defer_until")]
        for key in defer_keys:
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {phase[key]}")
        lines.append("")
    git = status.get("git") or {}
    lines.append("## Local")
    branch = git.get("branch", "?")
    clean = git.get("clean", True)
    dirty_count = git.get("dirty_count", 0)
    tree = "clean" if clean else f"dirty ({dirty_count} paths)"
    lines.append(f"- **Git:** `{branch}` — working tree {tree}")
    gpu = status.get("gpu_lock") or {}
    if gpu.get("held"):
        lines.append(
            "- **GPU lock:** held — "
            f"`{gpu.get('job', '?')}` pid={gpu.get('pid', '?')} "
            f"since {gpu.get('started_at', '?')}"
        )
    else:
        lines.append("- **GPU lock:** free")
    manifest = status.get("manifest")
    if manifest:
        if manifest.get("error"):
            lines.append(f"- **Manifest:** `{manifest.get('path')}` — unreadable")
        else:
            lines.append(
                "- **Manifest:** "
                f"`{manifest.get('path')}` — "
                f"{manifest.get('train_rows')}/{manifest.get('val_rows')} rows, "
                f"{manifest.get('train_songs')}/{manifest.get('val_songs')} songs "
                f"(created {manifest.get('created_at', '?')})"
            )
    latest = status.get("latest_exp")
    if latest:
        lines.append(f"- **Latest EXP (index):** {latest}")
    missing = status.get("missing_prerequisites") or []
    if missing:
        lines.append("- **Missing prerequisites:**")
        for path in missing:
            lines.append(f"  - `{path}`")
    else:
        lines.append("- **Missing prerequisites:** none checked")
    recent = status.get("recent_logs") or []
    if recent:
        lines.append("- **Recent logs:**")
        for row in recent:
            lines.append(f"  - `{row['path']}` ({row['modified_iso']})")
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project orientation status (read-only)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of markdown",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    args = _build_parser().parse_args(argv)
    status = collect_status()
    if args.json:
        # JSON cannot serialize mixed types from pathlib mtime float cleanly in nested
        print(json.dumps(status, indent=2, default=str))
    else:
        print(_format_markdown(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
