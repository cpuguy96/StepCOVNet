"""Execute one agent-planned AR tide iteration, or loop overnight with auto-planning.

Scratch-only: ``overnight_planner`` proposes each recipe (no warm-start).

Usage (repo root):

    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_overnight.py --hours 7
    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_overnight.py --once
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_ITER_PKG = Path(__file__).resolve().parent
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

import config_builder  # noqa: E402
from overnight_planner import plan_next_experiment  # noqa: E402
from session_brief import (  # noqa: E402
    NEXT_EXPERIMENT_PATH,
    build_brief,
    format_brief_text,
)
from training_lock import assert_gpu_training_available  # noqa: E402

PY = REPO / "venv" / "Scripts" / "python.exe"
RUN_EXP = REPO / "scripts" / "ar_tide_iter" / "run_exp.py"
APPLIED_DIR = REPO / "logs" / "ar_tide_iter" / "applied_plans"
PLANS_DIR = REPO / "logs" / "ar_tide_iter" / "auto_plans"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the planned experiment in next_experiment.json",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="Loop until deadline: plan scratch experiments and run each (default 0=off)",
    )
    parser.add_argument(
        "--brief-only",
        action="store_true",
        help="Print session brief and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to run_exp (GPU busy override)",
    )
    return parser.parse_args()


def _validate_plan(plan: dict) -> dict:
    for key in ("id", "notes"):
        if key not in plan:
            msg = f"next_experiment.json missing required key: {key}"
            raise ValueError(msg)
    if "run" not in plan and "model" not in plan and "dataset" not in plan:
        raise ValueError(
            "next_experiment.json needs at least one override block: run, model, or dataset",
        )
    return config_builder.prepare_experiment_spec(plan)


def _archive_plan(plan: dict, *, subdir: Path = APPLIED_DIR) -> Path:
    subdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = subdir / f"{plan['id']}.{stamp}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return out


def _run_planned(exp_id: str, notes: str, *, force: bool) -> int:
    cmd = [str(PY), str(RUN_EXP), "--id", exp_id, "--notes", notes]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=REPO)
    return int(proc.returncode)


def execute_plan(plan: dict, *, force: bool) -> int:
    """Register, archive, and run one experiment plan."""
    try:
        spec = _validate_plan(plan)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    exp_id = spec["id"]
    notes = spec["notes"]
    print(f"[plan] {exp_id}: {notes}")

    config_builder.register_adaptive_experiment(exp_id, notes=notes, plan=plan)
    archived = _archive_plan(plan)
    if NEXT_EXPERIMENT_PATH.is_file():
        NEXT_EXPERIMENT_PATH.unlink(missing_ok=True)
    print(f"archived plan: {archived.relative_to(REPO)}")

    try:
        assert_gpu_training_available(exp_id=exp_id, force=force)
    except RuntimeError as exc:
        print(f"GPU busy: {exc}", file=sys.stderr)
        return 1

    return _run_planned(exp_id, notes, force=force)


def run_hours_loop(hours: float, *, force: bool) -> int:
    """Plan and run scratch experiments until deadline or 634/634 free-run."""
    deadline = datetime.now() + timedelta(hours=hours)
    print(f"Overnight scratch loop — budget {hours} h, deadline {deadline.isoformat()}")
    run_index = 0
    last_exit = 0

    while datetime.now() < deadline:
        while datetime.now() < deadline:
            try:
                assert_gpu_training_available(exp_id="overnight", force=force)
                break
            except RuntimeError as exc:
                print(f"GPU busy — waiting: {exc}")
                time.sleep(30)
        else:
            break

        plan = plan_next_experiment(run_index=run_index)
        if plan is None:
            print("Session pass (634/634 free-run) — stopping overnight loop.")
            return 0

        run_index += 1
        remaining = deadline - datetime.now()
        print(
            f"\n=== overnight run {run_index} | {plan['id']} | "
            f"{remaining.total_seconds() / 3600:.1f} h left ===",
        )
        _archive_plan(plan, subdir=PLANS_DIR)
        last_exit = execute_plan(plan, force=force)
        print()
        print(format_brief_text(build_brief()))

        brief = build_brief()
        best = brief.get("session_best") or {}
        if best.get("free_run_matched") == 634:
            print("634/634 free-run — overnight success.")
            return 0

        if datetime.now() >= deadline:
            break

    print(f"Overnight deadline reached after {run_index} run(s).")
    return last_exit


def main() -> int:
    args = _parse_args()
    if args.brief_only:
        print(format_brief_text(build_brief()))
        return 0

    if args.hours > 0:
        return run_hours_loop(args.hours, force=args.force)

    if not NEXT_EXPERIMENT_PATH.is_file():
        print(format_brief_text(build_brief()), file=sys.stderr)
        print(
            f"\nNo plan at {NEXT_EXPERIMENT_PATH.relative_to(REPO)}.",
            file=sys.stderr,
        )
        print(
            "Write next_experiment.json, use --once, or run unattended: --hours 7",
            file=sys.stderr,
        )
        return 2

    plan = json.loads(NEXT_EXPERIMENT_PATH.read_text(encoding="utf-8"))
    exit_code = execute_plan(plan, force=args.force)
    print()
    print(format_brief_text(build_brief()))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
