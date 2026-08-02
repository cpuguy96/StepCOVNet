#!/usr/bin/env python3
"""Launch TensorBoard for cross-run metric comparison.

Discovers training log roots under ``callbacks/``, ``experiments/``, and
``e2e_mert/``, then starts TensorBoard with a scope wide enough to overlay
reruns and related experiments.

Usage (repo root, project venv):

    python scripts/tensorboard_compare.py --list
    python scripts/tensorboard_compare.py --root callbacks/ar
    python scripts/tensorboard_compare.py --preset ladder
    python scripts/tensorboard_compare.py --config configs/ar/ladder_10t_50v.json \\
        configs/ar/ladder_50t_50v.json
    python scripts/tensorboard_compare.py --filter scale_50t --print-cmd
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SCAN_ROOTS = (
    REPO / "callbacks",
    REPO / "experiments",
    REPO / "e2e_mert",
)
TFEVENTS_GLOB = "events.out.tfevents.*"

PRESET_FILTERS: dict[str, re.Pattern[str]] = {
    "ladder": re.compile(r"ladder_\d+t", re.IGNORECASE),
    "scale": re.compile(r"scale_\d+t", re.IGNORECASE),
    "dense": re.compile(r"dense|final_data_dense", re.IGNORECASE),
    "overfit": re.compile(r"overfit", re.IGNORECASE),
    "ar": re.compile(r"^callbacks[/\\]ar[/\\]", re.IGNORECASE),
}


def _has_tfevents(directory: pathlib.Path) -> bool:
    """Return True when ``directory`` contains TensorBoard event files."""
    return any(directory.glob(TFEVENTS_GLOB)) or any(
        directory.rglob(TFEVENTS_GLOB),
    )


def _count_runs(logs_dir: pathlib.Path) -> int:
    """Count timestamped run folders under a callback ``logs/`` directory."""
    if not logs_dir.is_dir():
        return 0
    count = 0
    for child in logs_dir.iterdir():
        if child.is_dir() and _has_tfevents(child):
            count += 1
    return count


def _discover_log_groups(
    scan_roots: tuple[pathlib.Path, ...],
) -> list[tuple[str, pathlib.Path, int]]:
    """Return ``(label, logs_dir, run_count)`` for every callback tree with data."""
    groups: dict[str, tuple[pathlib.Path, int]] = {}
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for logs_dir in scan_root.rglob("logs"):
            if not logs_dir.is_dir() or not _has_tfevents(logs_dir):
                continue
            try:
                callback_root = logs_dir.parent.relative_to(REPO)
            except ValueError:
                continue
            key = callback_root.as_posix()
            run_count = _count_runs(logs_dir)
            existing = groups.get(key)
            if existing is None or run_count > existing[1]:
                groups[key] = (logs_dir, run_count)
    return sorted(
        (_label_for_callback_root(key), logs_dir, run_count)
        for key, (logs_dir, run_count) in groups.items()
    )


def _label_for_callback_root(rel_callback_root: str) -> str:
    """Build a short, TensorBoard-safe label from a callback root path."""
    label = rel_callback_root.replace("\\", "/")
    if label.startswith("callbacks/"):
        label = label[len("callbacks/") :]
    label = re.sub(r"[^A-Za-z0-9/_-]+", "_", label)
    return label or "run"


def _sanitize_tb_label(label: str) -> str:
    """Remove characters that break ``--logdir_spec`` parsing."""
    return re.sub(r"[:,]", "_", label)


def _read_callback_logdir(config_path: pathlib.Path) -> pathlib.Path | None:
    """Resolve ``run.callback_root_dir`` from a training JSON config."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    callback_root = str(data.get("run", {}).get("callback_root_dir", "")).strip()
    if not callback_root:
        return None
    logs_dir = (REPO / callback_root / "logs").resolve()
    if not logs_dir.is_dir() or not _has_tfevents(logs_dir):
        return None
    return logs_dir


def _tensorboard_executable() -> pathlib.Path:
    """Return the project-venv TensorBoard executable."""
    if sys.platform == "win32":
        candidate = REPO / "venv" / "Scripts" / "tensorboard.exe"
    else:
        candidate = REPO / "venv" / "bin" / "tensorboard"
    if candidate.is_file():
        return candidate
    found = shutil.which("tensorboard")
    if found:
        return pathlib.Path(found)
    raise FileNotFoundError(
        "tensorboard not found — install project venv or add tensorboard to PATH",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch TensorBoard scoped for cross-run comparison. "
            "Use --root for one recursive tree, or --preset/--filter/--config "
            "to group named experiments."
        ),
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help=(
            "Single recursive logdir (repeatable). Example: callbacks/ar shows "
            "every AR experiment and every rerun under it."
        ),
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help=(
            "Explicit callback log root to include as a labeled group "
            "(path to .../logs or its callback parent)."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Training JSON config; uses run.callback_root_dir/logs.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_FILTERS),
        help="Filter discovered groups by common experiment families.",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Regex applied to discovered callback-root paths.",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="Extra directory to scan (default: callbacks, experiments, e2e_mert).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered log groups and exit.",
    )
    parser.add_argument(
        "--print-cmd",
        action="store_true",
        help="Print the tensorboard command without launching.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="TensorBoard port (default: 6006).",
    )
    parser.add_argument(
        "--reload-interval",
        type=int,
        default=30,
        help="Seconds between TensorBoard reloads (default: 30).",
    )
    return parser


def _resolve_group_path(raw: str) -> pathlib.Path | None:
    """Normalize a user path to an existing ``.../logs`` directory."""
    path = (REPO / raw).resolve()
    if path.name != "logs" and (path / "logs").is_dir():
        path = path / "logs"
    if not path.is_dir() or not _has_tfevents(path):
        return None
    return path


def _select_groups(args: argparse.Namespace) -> list[tuple[str, pathlib.Path, int]]:
    """Resolve the log groups to show based on CLI arguments."""
    selected: list[tuple[str, pathlib.Path, int]] = []

    for config_arg in args.config:
        config_path = (REPO / config_arg).resolve()
        if not config_path.is_file():
            raise SystemExit(f"Config not found: {config_arg}")
        logs_dir = _read_callback_logdir(config_path)
        if logs_dir is None:
            raise SystemExit(
                f"No TensorBoard logs for config callback_root_dir: {config_arg}",
            )
        label = _label_for_callback_root(
            logs_dir.parent.relative_to(REPO).as_posix(),
        )
        selected.append((label, logs_dir, _count_runs(logs_dir)))

    for group_arg in args.group:
        logs_dir = _resolve_group_path(group_arg)
        if logs_dir is None:
            raise SystemExit(f"No TensorBoard logs at: {group_arg}")
        label = _label_for_callback_root(
            logs_dir.parent.relative_to(REPO).as_posix(),
        )
        selected.append((label, logs_dir, _count_runs(logs_dir)))

    discover = bool(args.preset or args.filter or args.list)
    if discover:
        scan_roots = (
            tuple(pathlib.Path(item).resolve() for item in (args.scan_root or []))
            or DEFAULT_SCAN_ROOTS
        )
        discovered = _discover_log_groups(scan_roots)
        pattern = re.compile(args.filter) if args.filter else None
        preset_pattern = PRESET_FILTERS.get(args.preset or "")
        for label, logs_dir, run_count in discovered:
            rel = logs_dir.parent.relative_to(REPO).as_posix()
            if preset_pattern and not preset_pattern.search(rel):
                continue
            if pattern and not pattern.search(rel):
                continue
            selected.append((label, logs_dir, run_count))

    deduped: dict[str, tuple[pathlib.Path, int]] = {}
    for label, logs_dir, run_count in selected:
        key = logs_dir.as_posix()
        deduped[key] = (label, logs_dir, run_count)
    return sorted(deduped.values(), key=lambda item: item[1].as_posix())


def _build_tensorboard_argv(
    args: argparse.Namespace,
    groups: list[tuple[str, pathlib.Path, int]],
) -> list[str]:
    """Build the tensorboard command line."""
    tb_exe = _tensorboard_executable()
    argv = [
        str(tb_exe),
        "--port",
        str(args.port),
        "--reload_interval",
        str(args.reload_interval),
    ]

    roots = [((REPO / raw).resolve()) for raw in args.root]
    if roots:
        if len(roots) == 1:
            argv.extend(["--logdir", roots[0].as_posix()])
        else:
            spec = ",".join(
                f"{_sanitize_tb_label(pathlib.Path(raw).as_posix())}:{path.as_posix()}"
                for raw, path in zip(args.root, roots, strict=True)
            )
            argv.extend(["--logdir_spec", spec])
        return argv

    if not groups:
        raise SystemExit(
            "No TensorBoard logs matched. Try --list, broaden --filter, or pass --root.",
        )

    if len(groups) == 1:
        argv.extend(["--logdir", groups[0][1].as_posix()])
        return argv

    spec_parts: list[str] = []
    used_labels: set[str] = set()
    for label, logs_dir, _run_count in groups:
        safe_label = _sanitize_tb_label(label)
        if safe_label in used_labels:
            suffix = 2
            candidate = f"{safe_label}_{suffix}"
            while candidate in used_labels:
                suffix += 1
                candidate = f"{safe_label}_{suffix}"
            safe_label = candidate
        used_labels.add(safe_label)
        spec_parts.append(f"{safe_label}:{logs_dir.as_posix()}")
    argv.extend(["--logdir_spec", ",".join(spec_parts)])
    return argv


def _print_group_list(groups: list[tuple[str, pathlib.Path, int]]) -> None:
    """Print discovered groups in a stable, human-readable table."""
    if not groups:
        print("No TensorBoard log groups found.")
        return
    print(f"{'runs':>4}  {'label':<40}  logs")
    print(f"{'----':>4}  {'-----':<40}  ----")
    for label, logs_dir, run_count in groups:
        rel_logs = logs_dir.relative_to(REPO).as_posix()
        print(f"{run_count:>4}  {label:<40}  {rel_logs}")


def _has_launch_scope(args: argparse.Namespace) -> bool:
    """Return True when the user selected an explicit TensorBoard scope."""
    return bool(
        args.root or args.group or args.config or args.preset or args.filter,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the TensorBoard compare CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not _has_launch_scope(args) and not args.list:
        parser.error(
            "choose a scope: --root, --preset, --filter, --config, --group, or --list",
        )

    groups = _select_groups(args)

    if args.list:
        _print_group_list(groups)
        return 0

    command = _build_tensorboard_argv(args, groups)
    if args.root:
        print("TensorBoard root mode:")
    else:
        print("TensorBoard groups:")
        _print_group_list(groups)
    print()
    print("Command:")
    print(subprocess.list2cmdline(command))
    print(f"URL: http://localhost:{args.port}/")

    if args.print_cmd:
        return 0

    print()
    print("Starting TensorBoard — enable multiple runs in the left sidebar to compare.")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
