"""Structured run summaries for agent autoresearch loops."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ITER_PKG = Path(__file__).resolve().parent
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

from results_history import (  # noqa: E402
    REPO,
    ResultRecord,
    load_results,
    next_iter_id,
    parse_free_run,
    parse_teacher,
)
from session_brief import GOAL_MATCHED, build_brief  # noqa: E402

SUMMARY_MARKER = "=== AUTORESEARCH_SUMMARY ==="

OUTCOME_GOAL = "goal_passed"
OUTCOME_TEACHER = "teacher_gate"
OUTCOME_FREE_RUN = "free_run_incomplete"
OUTCOME_INFRA = "infra_failure"
OUTCOME_NO_RESULT = "no_result"


def _latest_result(exp_id: str, *, repo: Path = REPO) -> ResultRecord | None:
    for record in reversed(load_results(repo=repo)):
        if record.id == exp_id:
            return record
    return None


def _load_result_row(exp_id: str, *, repo: Path = REPO) -> dict[str, Any] | None:
    results_path = repo / "logs" / "ar_tide_iter" / "results.jsonl"
    if not results_path.is_file():
        return None
    for line in reversed(results_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("id")) == exp_id:
            return row
    return None


def classify_outcome(
    *,
    raw_exit_code: int,
    record: ResultRecord | None,
    row: dict[str, Any] | None,
) -> str:
    if row and bool(row.get("passed")):
        return OUTCOME_GOAL
    if record is not None and record.passed:
        return OUTCOME_GOAL
    if record is None and raw_exit_code != 0:
        return OUTCOME_INFRA
    if record is None:
        return OUTCOME_NO_RESULT
    if record.teacher_gate_failed:
        return OUTCOME_TEACHER
    train_exit = row.get("train_exit") if row else record.train_exit
    if train_exit not in (0, "0", "skipped"):
        return OUTCOME_INFRA
    if row and row.get("error") and not record.teacher_gate_failed:
        return OUTCOME_INFRA
    if record.matched is not None:
        return OUTCOME_FREE_RUN
    if raw_exit_code != 0:
        return OUTCOME_INFRA
    return OUTCOME_TEACHER


def remap_exit_code(*, outcome: str, raw_exit_code: int) -> int:
    """Autoresearch: distinguish infra failures from goal-not-met completions."""
    if outcome in (OUTCOME_INFRA, OUTCOME_NO_RESULT):
        return 1 if raw_exit_code == 0 else raw_exit_code
    return 0


def build_run_summary(
    exp_id: str,
    *,
    raw_exit_code: int,
    repo: Path = REPO,
    mode: str = "once",
) -> dict[str, Any]:
    record = _latest_result(exp_id, repo=repo)
    row = _load_result_row(exp_id, repo=repo)
    outcome = classify_outcome(
        raw_exit_code=raw_exit_code,
        record=record,
        row=row,
    )
    exit_code = remap_exit_code(outcome=outcome, raw_exit_code=raw_exit_code)

    teacher_text = str(row.get("teacher", "")) if row else ""
    teacher_pair = parse_teacher(teacher_text) if teacher_text else None
    free_text = str(row.get("free_run", "")) if row else ""
    free_pair = parse_free_run(free_text) if free_text else None
    if (
        record is not None
        and free_pair is None
        and record.matched is not None
        and record.denom is not None
    ):
        free_pair = (record.matched, record.denom)

    brief = build_brief(repo=repo)
    session_best = brief.get("session_best")

    summary: dict[str, Any] = {
        "mode": mode,
        "exp_id": exp_id,
        "raw_exit_code": raw_exit_code,
        "exit_code": exit_code,
        "outcome": outcome,
        "goal_passed": outcome == OUTCOME_GOAL,
        "goal": f"free_run_ordered_match {GOAL_MATCHED}/{GOAL_MATCHED} @ 20ms",
        "teacher": teacher_text or None,
        "teacher_matched": teacher_pair[0] if teacher_pair else None,
        "teacher_denom": teacher_pair[1] if teacher_pair else None,
        "free_run": free_text or None,
        "free_run_matched": free_pair[0] if free_pair else None,
        "free_run_denom": free_pair[1] if free_pair else None,
        "train_exit": record.train_exit if record else None,
        "notes": record.notes if record else None,
        "error": row.get("error") if row else None,
        "suggested_next_id": brief.get("suggested_next_id")
        or next_iter_id(
            load_results(repo=repo),
            repo=repo,
        ),
        "session_best": session_best,
        "stop_reason": None,
    }
    return summary


def emit_summary(summary: dict[str, Any], *, as_json_only: bool = False) -> None:
    text = json.dumps(summary, indent=2)
    if as_json_only:
        print(text)
        return
    print()
    print(SUMMARY_MARKER)
    print(text)
