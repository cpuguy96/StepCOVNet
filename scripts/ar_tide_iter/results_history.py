"""Load and parse AR tide iteration results for session briefs."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS_JSONL = REPO / "logs" / "ar_tide_iter" / "results.jsonl"

_MATCHED_DENOM_RE = re.compile(
    r"^(?P<matched>\d+)/(?P<denom>\d+)\s+\((?P<rate>[0-9.]+)\)$",
)
_ITER_ID_RE = re.compile(r"^iter(\d+)$")


@dataclass(frozen=True)
class ResultRecord:
    """One logged train/eval attempt."""

    id: str
    timestamp: str
    notes: str
    model_path: str
    config: str
    train_exit: int | str
    matched: int | None
    denom: int | None
    teacher_gate_failed: bool
    passed: bool
    run: dict[str, object]

    @property
    def score(self) -> float:
        """Sort key: higher is better; teacher failures sink to the bottom."""
        if self.passed:
            return 10_000.0
        if self.teacher_gate_failed:
            return -1.0
        if self.matched is None:
            return -2.0
        return float(self.matched)


def parse_matched_denom(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    match = _MATCHED_DENOM_RE.match(str(text).strip())
    if not match:
        return None
    return int(match.group("matched")), int(match.group("denom"))


def parse_free_run(text: str | None) -> tuple[int, int] | None:
    return parse_matched_denom(text)


def parse_teacher(text: str | None) -> tuple[int, int] | None:
    return parse_matched_denom(text)


def _load_config_block(config_rel: str, repo: pathlib.Path) -> dict[str, object]:
    path = pathlib.Path(config_rel)
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        return {}
    return cfg


def _load_run_block(config_rel: str, repo: pathlib.Path) -> dict[str, object]:
    run = _load_config_block(config_rel, repo).get("run", {})
    if not isinstance(run, dict):
        return {}
    return dict(run)


def load_results(
    path: pathlib.Path | None = None,
    *,
    repo: pathlib.Path = REPO,
) -> list[ResultRecord]:
    """Return all results in file order."""
    results_path = path or (repo / "logs" / "ar_tide_iter" / "results.jsonl")
    if not results_path.is_file():
        return []

    records: list[ResultRecord] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parsed = parse_free_run(row.get("free_run"))
        matched = row.get("free_run_matched")
        denom = row.get("free_run_denom")
        if matched is None and parsed is not None:
            matched, denom = parsed
        matched = int(matched) if isinstance(matched, (int, float)) else None
        denom = int(denom) if isinstance(denom, (int, float)) else None
        teacher_gate_failed = bool(row.get("teacher_gate_failed")) or str(
            row.get("error", ""),
        ).startswith("teacher metrics")
        records.append(
            ResultRecord(
                id=str(row["id"]),
                timestamp=str(row.get("timestamp", "")),
                notes=str(row.get("notes", "")),
                model_path=str(row.get("model_path", "")).replace("\\", "/"),
                config=str(row.get("config", "")).replace("\\", "/"),
                train_exit=row.get("train_exit", "skipped"),
                matched=matched,
                denom=denom,
                teacher_gate_failed=teacher_gate_failed,
                passed=bool(row.get("passed")),
                run=_load_run_block(str(row.get("config", "")), repo),
            ),
        )
    return records


def max_iter_number(records: list[ResultRecord]) -> int:
    best = 0
    for record in records:
        match = _ITER_ID_RE.match(record.id)
        if match:
            best = max(best, int(match.group(1)))
    return best


def _max_registered_iter(repo: pathlib.Path) -> int:
    exp_path = repo / "scripts" / "ar_tide_iter" / "experiments.json"
    if not exp_path.is_file():
        return 0
    best = 0
    for spec in json.loads(exp_path.read_text(encoding="utf-8")):
        match = _ITER_ID_RE.match(str(spec.get("id", "")))
        if match:
            best = max(best, int(match.group(1)))
    return best


def next_iter_id(
    records: list[ResultRecord],
    *,
    repo: pathlib.Path = REPO,
) -> str:
    best = max(max_iter_number(records), _max_registered_iter(repo))
    return f"iter{best + 1}"
