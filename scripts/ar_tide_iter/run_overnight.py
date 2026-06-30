"""Execute one agent-planned AR tide iteration, or loop with auto-planning.

Scratch-only: ``overnight_planner`` proposes each recipe (no warm-start).

Usage (repo root):

    # Agent autoresearch (recommended — no knob lattice):
    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --once
    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --hours 6

    # Unattended lattice planner (not for Cursor autoresearch skill):
    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_overnight.py --hours 7 --allow-planner

Exit codes (``--autoresearch`` remaps goal-not-met to 0):

    0 — run completed (or goal passed)
    1 — plan/GPU/train infra failure
    2 — missing ``next_experiment.json``
    3 — ``--hours`` without ``--allow-planner`` or ``--autoresearch``
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
from run_summary import build_run_summary, emit_summary  # noqa: E402
from session_brief import (  # noqa: E402
    NEXT_EXPERIMENT_PATH,
    build_brief,
    format_brief_text,
)
from training_lock import (  # noqa: E402
    assert_gpu_training_available,
    wait_gpu_training_available,
)

PY = REPO / "venv" / "Scripts" / "python.exe"
RUN_EXP = REPO / "scripts" / "ar_tide_iter" / "run_exp.py"
APPLIED_DIR = REPO / "logs" / "ar_tide_iter" / "applied_plans"
PLANS_DIR = REPO / "logs" / "ar_tide_iter" / "auto_plans"
PLAN_POLL_SEC = 10.0

EXIT_OK = 0
EXIT_INFRA = 1
EXIT_NO_PLAN = 2
EXIT_MODE = 3


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
        help="Loop until deadline (see --autoresearch vs --allow-planner)",
    )
    parser.add_argument(
        "--autoresearch",
        action="store_true",
        help="Agent loop: JSON summary, remapped exit codes, no overnight_planner",
    )
    parser.add_argument(
        "--allow-planner",
        action="store_true",
        help="With --hours: use overnight_planner lattice (not for autoresearch skill)",
    )
    parser.add_argument(
        "--plan-wait",
        type=float,
        default=600.0,
        help="Autoresearch --hours: seconds to wait for next_experiment.json between runs",
    )
    parser.add_argument(
        "--brief-only",
        action="store_true",
        help="Print session brief and exit",
    )
    parser.add_argument(
        "--brief",
        choices=("text", "json", "none"),
        default="text",
        help="Session brief format after each run (default: text)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit AUTORESEARCH_SUMMARY JSON (implies useful exit codes with --autoresearch)",
    )
    parser.add_argument(
        "--wait-gpu",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Poll until GPU lock clears instead of failing immediately",
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


def _gpu_ready(*, exp_id: str, force: bool, wait_gpu: float) -> None:
    if wait_gpu > 0:
        wait_gpu_training_available(
            exp_id=exp_id,
            timeout_sec=wait_gpu,
            force=force,
        )
        return
    assert_gpu_training_available(exp_id=exp_id, force=force)


def _run_planned(exp_id: str, notes: str, *, force: bool) -> int:
    cmd = [str(PY), str(RUN_EXP), "--id", exp_id, "--notes", notes]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=REPO)
    return int(proc.returncode)


def execute_plan(
    plan: dict,
    *,
    force: bool,
    wait_gpu: float = 0.0,
) -> tuple[int, str]:
    """Register, archive, and run one experiment plan. Returns (exit_code, exp_id)."""
    try:
        spec = _validate_plan(plan)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INFRA, str(plan.get("id", "?"))

    exp_id = spec["id"]
    notes = spec["notes"]
    print(f"[plan] {exp_id}: {notes}")

    config_builder.register_adaptive_experiment(exp_id, notes=notes, plan=plan)
    archived = _archive_plan(plan)
    if NEXT_EXPERIMENT_PATH.is_file():
        NEXT_EXPERIMENT_PATH.unlink(missing_ok=True)
    print(f"archived plan: {archived.relative_to(REPO)}")

    try:
        _gpu_ready(exp_id=exp_id, force=force, wait_gpu=wait_gpu)
    except RuntimeError as exc:
        print(f"GPU busy: {exc}", file=sys.stderr)
        return EXIT_INFRA, exp_id

    return _run_planned(exp_id, notes, force=force), exp_id


def _print_brief(brief_mode: str) -> None:
    brief = build_brief()
    if brief_mode == "none":
        return
    if brief_mode == "json":
        print(json.dumps(brief, indent=2))
        return
    print(format_brief_text(brief))


def _finish_run(
    exp_id: str,
    raw_exit: int,
    *,
    autoresearch: bool,
    json_out: bool,
    brief_mode: str,
    mode: str,
) -> int:
    exit_code = raw_exit
    if autoresearch or json_out:
        summary = build_run_summary(exp_id, raw_exit_code=raw_exit, mode=mode)
        emit_summary(
            summary,
            as_json_only=bool(json_out and brief_mode == "none" and not autoresearch),
        )
        if autoresearch:
            exit_code = summary["exit_code"]
    if brief_mode != "none":
        _print_brief(brief_mode)
    return exit_code


def _load_plan() -> dict | None:
    if not NEXT_EXPERIMENT_PATH.is_file():
        return None
    return json.loads(NEXT_EXPERIMENT_PATH.read_text(encoding="utf-8-sig"))


def _wait_for_plan(*, deadline: datetime, plan_wait_sec: float) -> dict | None:
    """Poll for next_experiment.json until timeout or session deadline."""
    wait_until = min(deadline, datetime.now() + timedelta(seconds=plan_wait_sec))
    while datetime.now() < wait_until:
        plan = _load_plan()
        if plan is not None:
            return plan
        remaining = (wait_until - datetime.now()).total_seconds()
        print(
            f"Waiting for {NEXT_EXPERIMENT_PATH.relative_to(REPO)} "
            f"({remaining:.0f}s left in plan-wait window)...",
        )
        time.sleep(PLAN_POLL_SEC)
    return None


def run_autoresearch_loop(
    hours: float,
    *,
    force: bool,
    wait_gpu: float,
    plan_wait_sec: float,
    brief_mode: str,
    json_out: bool,
) -> int:
    """Agent-driven loop: run plans from next_experiment.json until deadline or pass."""
    deadline = datetime.now() + timedelta(hours=hours)
    print(
        f"Autoresearch loop — budget {hours} h, deadline {deadline.isoformat()}",
    )
    print(f"Write plans to {NEXT_EXPERIMENT_PATH.relative_to(REPO)} between runs.")
    run_index = 0
    last_exit = EXIT_OK

    while datetime.now() < deadline:
        plan = _load_plan()
        if plan is None:
            plan = _wait_for_plan(deadline=deadline, plan_wait_sec=plan_wait_sec)
        if plan is None:
            summary = {
                "mode": "autoresearch",
                "stop_reason": "plan_wait_timeout",
                "suggested_next_id": build_brief().get("suggested_next_id"),
                "exit_code": EXIT_OK,
            }
            emit_summary(summary, as_json_only=False)
            print(
                "Plan wait window expired — write next_experiment.json and re-invoke."
            )
            return EXIT_OK

        run_index += 1
        remaining = deadline - datetime.now()
        print(
            f"\n=== autoresearch run {run_index} | {plan['id']} | "
            f"{remaining.total_seconds() / 3600:.1f} h left ===",
        )
        raw_exit, exp_id = execute_plan(plan, force=force, wait_gpu=wait_gpu)
        summary = build_run_summary(
            exp_id,
            raw_exit_code=raw_exit,
            mode="autoresearch",
        )
        if datetime.now() >= deadline:
            summary["stop_reason"] = "budget_exhausted"
        emit_summary(summary, as_json_only=False)
        if brief_mode != "none":
            _print_brief(brief_mode)
        last_exit = summary["exit_code"]

        if summary.get("goal_passed"):
            summary["stop_reason"] = "goal_passed"
            print("634/634 free-run — autoresearch success.")
            return EXIT_OK

        if datetime.now() >= deadline:
            break

    print(f"Autoresearch deadline reached after {run_index} run(s).")
    return last_exit


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
        last_exit, _exp_id = execute_plan(plan, force=force)
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
    autoresearch = args.autoresearch
    json_out = args.json or autoresearch

    if args.brief_only:
        _print_brief("json" if args.json else args.brief)
        return EXIT_OK

    if args.hours > 0:
        if autoresearch:
            return run_autoresearch_loop(
                args.hours,
                force=args.force,
                wait_gpu=args.wait_gpu,
                plan_wait_sec=args.plan_wait,
                brief_mode=args.brief,
                json_out=json_out,
            )
        if not args.allow_planner:
            print(
                "Refusing unattended --hours without --allow-planner.\n"
                "For Cursor autoresearch use:\n"
                "  run_overnight.py --autoresearch --hours N\n"
                "or per iteration:\n"
                "  run_overnight.py --autoresearch --once",
                file=sys.stderr,
            )
            return EXIT_MODE
        return run_hours_loop(args.hours, force=args.force)

    if not NEXT_EXPERIMENT_PATH.is_file():
        if autoresearch or args.json:
            brief = build_brief()
            summary = {
                "mode": "once",
                "stop_reason": "no_plan",
                "suggested_next_id": brief.get("suggested_next_id"),
                "exit_code": EXIT_NO_PLAN,
                "session_best": brief.get("session_best"),
            }
            emit_summary(
                summary,
                as_json_only=bool(
                    json_out and args.brief == "none" and not autoresearch
                ),
            )
        else:
            print(format_brief_text(build_brief()), file=sys.stderr)
            print(
                f"\nNo plan at {NEXT_EXPERIMENT_PATH.relative_to(REPO)}.",
                file=sys.stderr,
            )
            print(
                "Write next_experiment.json, use --autoresearch --once, "
                "or run --autoresearch --hours N",
                file=sys.stderr,
            )
        return EXIT_NO_PLAN

    if not args.once and not autoresearch:
        print(
            "Specify --once or --autoresearch to run the plan in next_experiment.json.",
            file=sys.stderr,
        )
        return EXIT_MODE

    plan = _load_plan()
    assert plan is not None
    raw_exit, exp_id = execute_plan(plan, force=args.force, wait_gpu=args.wait_gpu)
    return _finish_run(
        exp_id,
        raw_exit,
        autoresearch=autoresearch,
        json_out=json_out,
        brief_mode=args.brief,
        mode="autoresearch" if autoresearch else "once",
    )


if __name__ == "__main__":
    raise SystemExit(main())
