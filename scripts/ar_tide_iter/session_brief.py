"""Build a decision brief for the agent between AR tide iteration runs.

The agent reads this output, reasons about training/eval results, and writes
``logs/ar_tide_iter/next_experiment.json``. No Python code auto-picks knobs.
Config diffs vs champion and prior runs show what changed; metrics stay fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ITER_PKG = Path(__file__).resolve().parent
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

from recipe_diff import (  # noqa: E402
    CHAMPION_PATH,
    FIXED_EVAL_METRICS,
    collect_config_keys,
    diff_config,
    format_changes,
    load_champion,
    recipe_fingerprint,
    summarize_run_delta,
)
from results_history import (  # noqa: E402
    REPO,
    ResultRecord,
    load_results,
    next_iter_id,
)
from training_log import parse_train_log, train_log_path  # noqa: E402

ITER_DIR = REPO / "logs" / "ar_tide_iter"
NEXT_EXPERIMENT_PATH = ITER_DIR / "next_experiment.json"
GOAL_MATCHED = 634


def _load_snapshot_config(record: ResultRecord, repo: Path) -> dict[str, Any]:
    path = Path(record.config)
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        return {"run": dict(record.run)}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg if isinstance(cfg, dict) else {"run": dict(record.run)}


def _summarize_record(
    record: ResultRecord,
    *,
    repo: Path,
    champion_run: dict[str, object],
    previous: ResultRecord | None,
) -> dict[str, Any]:
    free_run = (
        f"{record.matched}/{record.denom}"
        if record.matched is not None and record.denom is not None
        else None
    )
    snapshot = _load_snapshot_config(record, repo)
    run = snapshot.get("run", record.run)
    if not isinstance(run, dict):
        run = record.run
    row: dict[str, Any] = {
        "id": record.id,
        "timestamp": record.timestamp,
        "notes": record.notes,
        "model_path": record.model_path,
        "recipe": recipe_fingerprint(run),
        "run_changes_vs_champion": summarize_run_delta(run, baseline=champion_run),
        "free_run": free_run,
        "teacher_gate_failed": record.teacher_gate_failed,
        "passed": record.passed,
        "train_exit": record.train_exit,
    }
    if previous is not None:
        prev_cfg = _load_snapshot_config(previous, repo)
        row["config_changes_vs_previous"] = diff_config(prev_cfg, snapshot)
    status = parse_train_log(train_log_path(record.id, 1))
    last_val = status.get("last_val")
    if last_val:
        row["last_val"] = last_val
    return row


def build_brief(
    *,
    repo: Path = REPO,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Structured session state for agent reasoning."""
    history = load_results(results_path, repo=repo)
    champion = load_champion()
    champion_run = champion.get("run", {})
    if not isinstance(champion_run, dict):
        champion_run = {}
    champion_run = dict(champion_run)

    snapshots = [_load_snapshot_config(record, repo) for record in history]
    config_keys = collect_config_keys(champion=champion, configs=snapshots)

    scored = [record for record in history if record.matched is not None]
    best = max(scored, key=lambda r: (r.matched or 0, r.timestamp)) if scored else None

    recent_rows: list[dict[str, Any]] = []
    for index, record in enumerate(history[-8:]):
        global_index = len(history) - len(history[-8:]) + index
        previous = history[global_index - 1] if global_index > 0 else None
        recent_rows.append(
            _summarize_record(
                record,
                repo=repo,
                champion_run=champion_run,
                previous=previous,
            ),
        )

    tried_recipes = []
    seen: set[str] = set()
    for record in history:
        run = record.run
        summary = recipe_fingerprint(run)
        if summary in seen:
            continue
        seen.add(summary)
        tried_recipes.append(
            {
                "recipe": summary,
                "best_free_run": max(
                    (
                        r.matched
                        for r in history
                        if recipe_fingerprint(r.run) == summary
                        and r.matched is not None
                    ),
                    default=None,
                ),
                "teacher_failures": sum(
                    1
                    for r in history
                    if recipe_fingerprint(r.run) == summary and r.teacher_gate_failed
                ),
            },
        )

    last_vs_best: dict[str, Any] | None = None
    if history and best is not None:
        last_cfg = _load_snapshot_config(history[-1], repo)
        best_cfg = _load_snapshot_config(best, repo)
        last_vs_best = {
            "last_id": history[-1].id,
            "best_id": best.id,
            "config_changes": diff_config(best_cfg, last_cfg),
        }

    return {
        "goal": f"free_run_ordered_match {GOAL_MATCHED}/{GOAL_MATCHED} @ 20ms",
        "suggested_next_id": next_iter_id(history, repo=repo),
        "next_experiment_path": "logs/ar_tide_iter/next_experiment.json",
        "champion_template": (
            str(CHAMPION_PATH.relative_to(repo)).replace("\\", "/")
            if CHAMPION_PATH.is_relative_to(repo)
            else "configs/ar/tide_overfit.json"
        ),
        "fixed_eval_metrics": FIXED_EVAL_METRICS,
        "config_keys_available": config_keys,
        "session_best": (
            {
                "id": best.id,
                "free_run_matched": best.matched,
                "model_path": best.model_path,
                "recipe": recipe_fingerprint(best.run),
                "notes": best.notes,
            }
            if best
            else None
        ),
        "last_run_vs_session_best": last_vs_best,
        "recent_runs": recent_rows,
        "tried_recipes": tried_recipes,
        "agent_instructions": [
            "Read this brief and docs/research/AR_TIDE_OVERFIT_ITER_LOG.md.",
            "Infer what changed between runs from config_changes_vs_previous and run_changes_vs_champion — no fixed knob list.",
            "Write next_experiment.json with id, notes, reasoning, and only the config overrides you want to change (run/model/dataset).",
            "Omit unchanged keys; new champion-template keys are valid knobs when code adds them.",
            "Do not change eval metrics or gates — compare runs on the fixed_eval_metrics only.",
            "Train from scratch every run — init_model_path is cheating and is stripped by the harness.",
            "Then: venv\\Scripts\\python.exe scripts/ar_tide_iter/run_overnight.py --once",
            "Do not use a fixed iter queue or let code auto-mutate hyperparameters.",
            "Run one training driver at a time; a second run_exp will refuse if gpu_training.lock is held.",
        ],
    }


def format_brief_text(brief: dict[str, Any]) -> str:
    lines = [
        "=== AR tide iteration — agent decision brief ===",
        f"Goal: {brief['goal']}",
        f"Suggested next id: {brief['suggested_next_id']}",
        f"Champion template: {brief.get('champion_template', 'configs/ar/tide_overfit.json')}",
        "",
    ]
    best = brief.get("session_best")
    if best:
        lines.append(
            "Session best: "
            f"{best['id']} — {best['free_run_matched']}/{GOAL_MATCHED} "
            f"({best['recipe']})"
        )
        lines.append(f"  checkpoint: {best['model_path']}")
    else:
        lines.append("Session best: (no scored runs yet)")
    lines.append("")
    lines.append("Fixed eval metrics (do not change between runs):")
    for key, value in brief.get("fixed_eval_metrics", {}).items():
        if isinstance(value, list):
            lines.append(f"  {key}: {', '.join(str(v) for v in value)}")
        else:
            lines.append(f"  {key}: {value}")
    lines.append("")
    last_vs_best = brief.get("last_run_vs_session_best")
    if last_vs_best:
        lines.append(
            f"Last run ({last_vs_best['last_id']}) vs session best "
            f"({last_vs_best['best_id']}):"
        )
        for section, changes in last_vs_best.get("config_changes", {}).items():
            lines.append(f"  [{section}] {format_changes(changes)}")
        lines.append("")
    lines.append("Recent runs (oldest to newest in list):")
    for row in brief.get("recent_runs", []):
        free = row.get("free_run") or "no free-run (teacher gate or error)"
        flag = " TEACHER_FAIL" if row.get("teacher_gate_failed") else ""
        lines.append(f"  {row['id']}: {free} | {row['recipe']}{flag}")
        prev = row.get("config_changes_vs_previous")
        if prev:
            for section, changes in prev.items():
                lines.append(f"    vs prev [{section}]: {format_changes(changes)}")
        if row.get("last_val"):
            val = row["last_val"]
            lines.append(
                "    last_val: gate={val_overfit_gate:.4f} ordered={val_ordered_onset_match:.4f} "
                "f1={val_event_onset_f1:.4f}".format(
                    val_overfit_gate=val.get("val_overfit_gate", 0.0),
                    val_ordered_onset_match=val.get("val_ordered_onset_match", 0.0),
                    val_event_onset_f1=val.get("val_event_onset_f1", 0.0),
                ),
            )
    lines.append("")
    lines.append(
        "Agent must write next experiment JSON (partial overrides OK), then --once:"
    )
    lines.append(f"  {brief['next_experiment_path']}")
    for step in brief.get("agent_instructions", []):
        lines.append(f"  - {step}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()
    brief = build_brief()
    if args.json:
        print(json.dumps(brief, indent=2))
    else:
        print(format_brief_text(brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
