"""Execute one agent-planned AR tide iteration, then surface the decision brief.

The agent chooses the next recipe (knobs, warm-start, hypothesis) by reading
``session_brief.py`` and writing ``logs/ar_tide_iter/next_experiment.json``.
This script does **not** auto-mutate hyperparameters.

Usage (repo root):

    venv\\Scripts\\python.exe scripts/ar_tide_iter/session_brief.py
    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_overnight.py --once
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_ITER_PKG = Path(__file__).resolve().parent
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

import config_builder  # noqa: E402
from session_brief import (  # noqa: E402
    NEXT_EXPERIMENT_PATH,
    build_brief,
    format_brief_text,
)
from training_lock import assert_gpu_training_available  # noqa: E402

PY = REPO / "venv" / "Scripts" / "python.exe"
RUN_EXP = REPO / "scripts" / "ar_tide_iter" / "run_exp.py"
APPLIED_DIR = REPO / "logs" / "ar_tide_iter" / "applied_plans"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the planned experiment in next_experiment.json (default)",
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


def _archive_plan(plan: dict) -> Path:
    APPLIED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = APPLIED_DIR / f"{plan['id']}.{stamp}.json"
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return out


def _run_planned(exp_id: str, notes: str, *, force: bool) -> int:
    cmd = [str(PY), str(RUN_EXP), "--id", exp_id, "--notes", notes]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, cwd=REPO)
    return int(proc.returncode)


def main() -> int:
    args = _parse_args()
    if args.brief_only:
        print(format_brief_text(build_brief()))
        return 0

    if not NEXT_EXPERIMENT_PATH.is_file():
        print(format_brief_text(build_brief()), file=sys.stderr)
        print(
            f"\nNo plan at {NEXT_EXPERIMENT_PATH.relative_to(REPO)}.",
            file=sys.stderr,
        )
        print(
            "Agent: read the brief, decide the next hypothesis, write next_experiment.json "
            "(see scripts/ar_tide_iter/next_experiment.example.json), then re-run --once.",
            file=sys.stderr,
        )
        return 2

    plan = json.loads(NEXT_EXPERIMENT_PATH.read_text(encoding="utf-8"))
    try:
        spec = _validate_plan(plan)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    exp_id = spec["id"]
    notes = spec["notes"]
    reasoning = plan.get("reasoning", "")
    if reasoning:
        print(f"[plan] {exp_id}: {reasoning}")

    config_builder.register_adaptive_experiment(exp_id, notes=notes, plan=plan)
    archived = _archive_plan(plan)
    NEXT_EXPERIMENT_PATH.unlink()
    print(f"archived plan: {archived.relative_to(REPO)}")

    try:
        assert_gpu_training_available(exp_id=exp_id, force=args.force)
    except RuntimeError as exc:
        print(f"GPU busy: {exc}", file=sys.stderr)
        return 1

    exit_code = _run_planned(exp_id, notes, force=args.force)
    print()
    print(format_brief_text(build_brief()))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
