"""Compare AR tide iteration configs without a fixed knob list.

The champion template plus historical snapshots define which keys exist.
New training features add keys to the template automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CHAMPION_PATH = REPO / "configs" / "ar" / "tide_overfit.json"

_CONFIG_SECTIONS = ("dataset", "model", "run")
_RUN_ARTIFACT_KEYS = frozenset({"model_output_dir", "callback_root_dir"})

# Eval / logging metrics held fixed so runs stay comparable.
FIXED_EVAL_METRICS = {
    "primary_gate": "free_run ordered_onset_match @ tolerance_sec (634/634 @ 20 ms)",
    "teacher_prerequisite": "teacher-fed ordered_onset_match, event_f1, token_accuracy all 1.0 before --ar_decode",
    "training_checkpoint_metric": "val_overfit_gate",
    "offline_eval_script": "scripts/debug_ar_onset_overfit.py --ar_decode --json-only",
    "reported_training_val_keys": [
        "val_overfit_gate",
        "val_ordered_onset_match",
        "val_event_onset_f1",
        "val_incremental_consistency_loss",
    ],
}


def load_champion() -> dict[str, Any]:
    return json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))


def champion_section(section: str) -> dict[str, Any]:
    block = load_champion().get(section, {})
    return dict(block) if isinstance(block, dict) else {}


def _normalize(value: object) -> object:
    if isinstance(value, float):
        return round(value, 12)
    return value


def diff_blocks(
    left: dict[str, object],
    right: dict[str, object],
    *,
    ignore_keys: frozenset[str] = frozenset(),
) -> dict[str, dict[str, object]]:
    """Return keys whose values differ between two flat config blocks."""
    keys = set(left) | set(right)
    changes: dict[str, dict[str, object]] = {}
    for key in sorted(keys):
        if key in ignore_keys:
            continue
        old = left.get(key)
        new = right.get(key)
        if _normalize(old) != _normalize(new):
            changes[key] = {"from": old, "to": new}
    return changes


def diff_config(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    ignore_run_keys: frozenset[str] = _RUN_ARTIFACT_KEYS,
) -> dict[str, dict[str, dict[str, object]]]:
    """Diff dataset/model/run sections between two full configs."""
    out: dict[str, dict[str, dict[str, object]]] = {}
    for section in _CONFIG_SECTIONS:
        left_block = left.get(section, {})
        right_block = right.get(section, {})
        if not isinstance(left_block, dict):
            left_block = {}
        if not isinstance(right_block, dict):
            right_block = {}
        ignore = ignore_run_keys if section == "run" else frozenset()
        section_diff = diff_blocks(left_block, right_block, ignore_keys=ignore)
        if section_diff:
            out[section] = section_diff
    return out


def collect_config_keys(
    *,
    champion: dict[str, Any] | None = None,
    configs: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Union of keys seen in champion + historical snapshots, per section."""
    base = champion if champion is not None else load_champion()
    keys: dict[str, set[str]] = {section: set() for section in _CONFIG_SECTIONS}
    for section in _CONFIG_SECTIONS:
        block = base.get(section, {})
        if isinstance(block, dict):
            keys[section].update(block)
    for cfg in configs or []:
        for section in _CONFIG_SECTIONS:
            block = cfg.get(section, {})
            if isinstance(block, dict):
                keys[section].update(block)
    return {section: sorted(keys[section]) for section in _CONFIG_SECTIONS}


def recipe_fingerprint(
    run: dict[str, object],
    *,
    ignore_keys: frozenset[str] = _RUN_ARTIFACT_KEYS,
) -> str:
    """Stable label for grouping runs by training recipe (not artifact paths)."""
    if not run:
        return "(config missing)"
    parts = []
    for key in sorted(run):
        if key in ignore_keys:
            continue
        parts.append(f"{key}={run[key]!r}")
    return " ".join(parts) if parts else "(champion run defaults)"


def format_changes(
    changes: dict[str, dict[str, object]],
    *,
    max_keys: int = 8,
) -> str:
    if not changes:
        return "(no changes)"
    lines: list[str] = []
    for index, (key, delta) in enumerate(sorted(changes.items())):
        if index >= max_keys:
            lines.append(f"... +{len(changes) - max_keys} more")
            break
        old = delta.get("from")
        new = delta.get("to")
        lines.append(f"{key}: {old!r} -> {new!r}")
    return "; ".join(lines)


def summarize_run_delta(
    run: dict[str, object],
    *,
    baseline: dict[str, object],
    ignore_keys: frozenset[str] = _RUN_ARTIFACT_KEYS,
) -> dict[str, dict[str, object]]:
    return diff_blocks(baseline, run, ignore_keys=ignore_keys)
