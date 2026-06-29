"""History-driven search for the next scratch-only AR tide overnight experiment.

Reads the full session (not just the last row), ranks runs by outcome, and
proposes a neighbor recipe from values seen in champion + historical configs.
No fixed hyperparameter ladder — new template keys become searchable automatically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ITER_PKG = Path(__file__).resolve().parent
import sys

if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))

from recipe_diff import (  # noqa: E402
    _CONFIG_SECTIONS,
    _RUN_ARTIFACT_KEYS,
    collect_config_keys,
    diff_blocks,
    load_champion,
    recipe_fingerprint,
)
from results_history import (  # noqa: E402
    REPO,
    RESULTS_JSONL,
    ResultRecord,
    load_results,
    next_iter_id,
    parse_free_run,
)

_ITER_ID_RE = re.compile(r"^iter(\d+)$")
_SCRATCH_ERA_MIN = 43

GOAL_MATCHED = 634
_TEACHER_RE = re.compile(
    r"ordered=(?P<matched>\d+)/(?P<denom>\d+)",
)
_IGNORE_RUN_KEYS = _RUN_ARTIFACT_KEYS | frozenset({"init_model_path"})
# Held fixed per overnight eval policy — never mutated by history search.
_PIN_RUN_KEYS = frozenset(
    {
        "epochs",
        "checkpoint_metric",
        "tolerance_sec",
    },
)


def _results_path(repo: Path) -> Path:
    if repo == REPO:
        return RESULTS_JSONL
    return repo / "logs" / "ar_tide_iter" / "results.jsonl"


def _load_rows(repo: Path) -> list[dict[str, Any]]:
    path = _results_path(repo)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_snapshot_config(record: ResultRecord, repo: Path) -> dict[str, Any]:
    path = Path(record.config)
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        return {"run": dict(record.run)}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg if isinstance(cfg, dict) else {"run": dict(record.run)}


def _teacher_matched(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    parsed = parse_free_run(str(row.get("teacher", "")))
    if parsed:
        return parsed[0]
    error = str(row.get("error", ""))
    match = _TEACHER_RE.search(error)
    if match:
        return int(match.group("matched"))
    return None


def _score_tuple(record: ResultRecord, row: dict[str, Any]) -> tuple[int, int, int]:
    """Higher sorts better: passed > free-run > teacher-only > missing."""
    if record.passed:
        return (4, GOAL_MATCHED, GOAL_MATCHED)
    teacher = _teacher_matched(row) or 0
    if record.teacher_gate_failed:
        return (1, teacher, 0)
    if record.matched is not None:
        return (3, int(record.matched), teacher)
    return (0, 0, 0)


def _best_row_per_id(
    records: list[ResultRecord],
    rows: list[dict[str, Any]],
) -> list[tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]]:
    by_id: dict[str, tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]] = {}
    for record, row in zip(records, rows, strict=False):
        score = _score_tuple(record, row)
        prev = by_id.get(record.id)
        if prev is None or score > prev[2]:
            by_id[record.id] = (record, row, score)
    return sorted(by_id.values(), key=lambda item: item[2], reverse=True)


def _section_block(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    block = cfg.get(section, {})
    return dict(block) if isinstance(block, dict) else {}


def _value_at(
    cfg: dict[str, Any], champion: dict[str, Any], section: str, key: str
) -> object:
    merged = _merged_config(cfg, champion)
    return _section_block(merged, section).get(key)


def _value_lattice(
    configs: list[dict[str, Any]],
    champion: dict[str, Any],
    *,
    keys_by_section: dict[str, list[str]],
) -> dict[tuple[str, str], set[object]]:
    lattice: dict[tuple[str, str], set[object]] = {}
    for section, keys in keys_by_section.items():
        for key in keys:
            if section == "run" and key in _IGNORE_RUN_KEYS | _PIN_RUN_KEYS:
                continue
            lattice[(section, key)] = set()
    sources = [champion, *configs]
    for cfg in sources:
        for section in _CONFIG_SECTIONS:
            block = _section_block(cfg, section)
            for key, value in block.items():
                if section == "run" and key in _IGNORE_RUN_KEYS | _PIN_RUN_KEYS:
                    continue
                lattice.setdefault((section, key), set()).add(value)
    return lattice


def _value_best_scores(
    ranked: list[tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]],
    *,
    repo: Path,
    champion: dict[str, Any],
) -> dict[tuple[str, str, object], tuple[int, int, int]]:
    scores: dict[tuple[str, str, object], tuple[int, int, int]] = {}
    for record, _row, score in ranked:
        cfg = _load_snapshot_config(record, repo)
        for section in _CONFIG_SECTIONS:
            keys = set(_section_block(cfg, section)) | set(
                _section_block(champion, section),
            )
            for key in keys:
                if section == "run" and key in _IGNORE_RUN_KEYS | _PIN_RUN_KEYS:
                    continue
                value = _value_at(cfg, champion, section, key)
                slot = (section, key, value)
                if slot not in scores or score > scores[slot]:
                    scores[slot] = score
    return scores


def _merged_config(cfg: dict[str, Any], champion: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(champion))
    for section in _CONFIG_SECTIONS:
        block = _section_block(cfg, section)
        if block:
            merged[section].update(block)
    return merged


def _overrides_vs_champion(
    full_cfg: dict[str, Any],
    champion: dict[str, Any],
) -> dict[str, dict[str, object]]:
    merged = _merged_config(full_cfg, champion)
    overrides: dict[str, dict[str, object]] = {}
    for section in _CONFIG_SECTIONS:
        block = _section_block(merged, section)
        base = _section_block(champion, section)
        ignore = _IGNORE_RUN_KEYS if section == "run" else frozenset()
        diff = diff_blocks(base, block, ignore_keys=ignore)
        if diff:
            overrides[section] = {
                key: delta["to"] for key, delta in diff.items() if "to" in delta
            }
    return overrides


def _full_cfg_from_overrides(
    overrides: dict[str, dict[str, object]],
    champion: dict[str, Any],
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(champion))
    for section, block in overrides.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            cfg[section] = {}
        cfg[section].update(block)
    return cfg


def _fingerprint_from_overrides(
    overrides: dict[str, dict[str, object]],
    champion: dict[str, Any],
) -> str:
    run = _full_cfg_from_overrides(overrides, champion).get("run", {})
    return recipe_fingerprint(run if isinstance(run, dict) else {})


def _tried_fingerprints(
    ranked: list[tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record, _row, _score in ranked:
        counts[recipe_fingerprint(record.run)] = (
            counts.get(recipe_fingerprint(record.run), 0) + 1
        )
    return counts


def _is_scratch_era(record: ResultRecord) -> bool:
    if "scratch" in record.notes.lower():
        return True
    match = _ITER_ID_RE.match(record.id)
    return bool(match and int(match.group(1)) >= _SCRATCH_ERA_MIN)


def _teacher_perfect_scratch(
    ranked: list[tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]],
) -> bool:
    for record, row, _score in ranked:
        if not _is_scratch_era(record):
            continue
        if _teacher_matched(row) == GOAL_MATCHED:
            return True
    return False


def _ranked_for_parent(
    ranked: list[tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]],
) -> list[tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]]:
    """Pick parent pool: scratch-era teacher progress first, then decode free-run."""
    if not _teacher_perfect_scratch(ranked):
        scratch_teacher = [
            item
            for item in ranked
            if item[0].teacher_gate_failed and _is_scratch_era(item[0])
        ]
        if scratch_teacher:
            return sorted(
                scratch_teacher,
                key=lambda item: (_teacher_matched(item[1]) or 0, item[2]),
                reverse=True,
            )
        teacher_failures = [
            item
            for item in ranked
            if item[0].teacher_gate_failed and _is_scratch_era(item[0])
        ]
        if teacher_failures:
            return sorted(
                teacher_failures,
                key=lambda item: (_teacher_matched(item[1]) or 0, item[2]),
                reverse=True,
            )
    scratch_scored = [
        item for item in ranked if _is_scratch_era(item[0]) and item[2][0] >= 3
    ]
    if scratch_scored:
        return scratch_scored
    return ranked


def _parent_pool(
    ranked: list[tuple[ResultRecord, dict[str, Any], tuple[int, int, int]]],
    *,
    repo: Path,
    limit: int = 5,
) -> list[tuple[ResultRecord, dict[str, Any], dict[str, Any]]]:
    seen: set[str] = set()
    parents: list[tuple[ResultRecord, dict[str, Any], dict[str, Any]]] = []
    for record, row, _score in _ranked_for_parent(ranked):
        fp = recipe_fingerprint(record.run)
        if fp in seen:
            continue
        seen.add(fp)
        parents.append((record, row, _load_snapshot_config(record, repo)))
        if len(parents) >= limit:
            break
    return parents


def _candidate_mutations(
    parent_cfg: dict[str, Any],
    *,
    champion: dict[str, Any],
    lattice: dict[tuple[str, str], set[object]],
    value_scores: dict[tuple[str, str, object], tuple[int, int, int]],
    tried: dict[str, int],
) -> list[tuple[dict[str, dict[str, object]], str]]:
    base_overrides = _overrides_vs_champion(parent_cfg, champion)
    candidates: list[
        tuple[dict[str, dict[str, object]], str, tuple[int, int, int], int]
    ] = []

    for (section, key), values in sorted(lattice.items()):
        if section == "run" and key in _PIN_RUN_KEYS:
            continue
        if len(values) < 2:
            continue
        current = _value_at(parent_cfg, champion, section, key)
        ranked_alts = sorted(
            values,
            key=lambda value: value_scores.get((section, key, value), (0, 0, 0)),
            reverse=True,
        )
        for alt in ranked_alts:
            if alt == current:
                continue
            overrides = json.loads(json.dumps(base_overrides))
            section_block = dict(overrides.get(section, {}))
            section_block[key] = alt
            overrides[section] = section_block
            fp = _fingerprint_from_overrides(overrides, champion)
            alt_score = value_scores.get((section, key, alt), (0, 0, 0))
            candidates.append(
                (overrides, fp, alt_score, tried.get(fp, 0)),
            )

    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda item: (item[3], -item[2][0], -item[2][1]),
    )
    return [(overrides, fp) for overrides, fp, _score, _tries in ranked]


def _apply_champion_pins(
    overrides: dict[str, dict[str, object]],
    champion: dict[str, Any],
) -> dict[str, dict[str, object]]:
    """Force budget/eval keys to champion values (e.g. epochs=200)."""
    champion_run = champion.get("run", {})
    if not isinstance(champion_run, dict):
        return overrides
    run = dict(overrides.get("run", {}))
    for key in _PIN_RUN_KEYS:
        if key in champion_run:
            run[key] = champion_run[key]
    if run:
        overrides = dict(overrides)
        overrides["run"] = run
    return overrides


def _format_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_notes(
    parent_id: str,
    overrides: dict[str, dict[str, object]],
) -> str:
    parts: list[str] = []
    for section in _CONFIG_SECTIONS:
        block = overrides.get(section, {})
        if not isinstance(block, dict):
            continue
        for key in sorted(block):
            parts.append(f"{key}={_format_value(block[key])}")
    if not parts:
        return f"from {parent_id}: retry"
    summary = ", ".join(parts[:5])
    if len(parts) > 5:
        summary += f" (+{len(parts) - 5})"
    return f"from {parent_id}: {summary}"


def plan_next_experiment(
    *,
    repo: Path = REPO,
    run_index: int = 0,
) -> dict[str, Any] | None:
    """Return the next scratch plan, or ``None`` if the session already passed."""
    results_path = _results_path(repo)
    history = load_results(results_path, repo=repo)
    if any(record.passed for record in history):
        return None

    rows = _load_rows(repo)
    if len(rows) != len(history):
        # Fallback: align by id when optional fields differ in length.
        rows_by_id = {}
        for row in rows:
            rows_by_id.setdefault(str(row["id"]), row)
        rows = [rows_by_id.get(record.id, {}) for record in history]

    champion = load_champion()
    ranked = _best_row_per_id(history, rows)
    if not ranked:
        exp_id = next_iter_id(history, repo=repo)
        return {
            "id": exp_id,
            "notes": "champion defaults",
            "run": {},
        }

    configs = [_load_snapshot_config(record, repo) for record, _row, _ in ranked]
    keys_by_section = collect_config_keys(champion=champion, configs=configs)
    lattice = _value_lattice(configs, champion, keys_by_section=keys_by_section)
    value_scores = _value_best_scores(ranked, repo=repo, champion=champion)
    tried = _tried_fingerprints(ranked)
    parents = _parent_pool(ranked, repo=repo)
    if not parents:
        parents = [
            (ranked[0][0], ranked[0][1], _load_snapshot_config(ranked[0][0], repo))
        ]

    parent_record, _parent_row, parent_cfg = parents[run_index % len(parents)]
    merged_parent = _merged_config(parent_cfg, champion)
    candidates = _candidate_mutations(
        merged_parent,
        champion=champion,
        lattice=lattice,
        value_scores=value_scores,
        tried=tried,
    )

    if candidates:
        untried = [item for item in candidates if item[1] not in tried]
        pool = untried if untried else candidates
        overrides, _fp = pool[run_index % len(pool)]
    else:
        overrides = _overrides_vs_champion(merged_parent, champion)

    overrides = _apply_champion_pins(overrides, champion)

    exp_id = next_iter_id(history, repo=repo)
    notes = _format_notes(
        parent_record.id,
        _overrides_vs_champion(
            _full_cfg_from_overrides(overrides, champion),
            champion,
        ),
    )

    plan: dict[str, Any] = {
        "id": exp_id,
        "notes": notes,
    }
    for section in _CONFIG_SECTIONS:
        if section in overrides:
            plan[section] = overrides[section]
    if "run" not in plan:
        plan["run"] = {}
    return plan
