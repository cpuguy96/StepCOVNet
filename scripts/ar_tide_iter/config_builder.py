"""Build full AR tide-overfit iteration configs from the champion template."""

from __future__ import annotations

import copy
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
CHAMPION = REPO / "configs" / "ar" / "tide_overfit.json"
EXPERIMENTS_PATH = pathlib.Path(__file__).resolve().parent / "experiments.json"
CONFIG_DIR = REPO / "logs" / "ar_tide_iter" / "configs"

_NULLABLE_RUN_KEYS = (
    "lambda_incremental_consistency",
    "incremental_consistency_max_steps",
)


def load_experiments() -> list[dict]:
    """Return experiment registry entries from ``experiments.json``."""
    return json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))


def get_experiment(exp_id: str) -> dict:
    """Return one registry entry by id."""
    for spec in load_experiments():
        if spec["id"] == exp_id:
            return spec
    known = ", ".join(spec["id"] for spec in load_experiments())
    msg = f"unknown experiment id {exp_id!r}; known: {known}"
    raise KeyError(msg)


def _strip_nones(run: dict) -> dict:
    return {key: value for key, value in run.items() if value is not None}


def build_config(spec: dict) -> dict:
    """Merge a registry spec onto the champion template."""
    base = json.loads(CHAMPION.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    run_updates = _strip_nones(spec["run"])
    if run_updates.get("init_model_path") is None:
        run_updates.pop("init_model_path", None)
        cfg["run"].pop("init_model_path", None)
    cfg["run"].update(run_updates)
    for key in _NULLABLE_RUN_KEYS:
        if key not in run_updates and key in spec["run"] and spec["run"][key] is None:
            cfg["run"].pop(key, None)
    return cfg


def config_path_for(exp_id: str, attempt: int = 1) -> pathlib.Path:
    """Gitignored path for a built config snapshot (one per attempt)."""
    if attempt <= 1:
        return CONFIG_DIR / f"{exp_id}.json"
    return CONFIG_DIR / f"{exp_id}.attempt{attempt}.json"


def latest_config_snapshot(
    exp_id: str, *, before_attempt: int | None = None
) -> pathlib.Path | None:
    """Return the newest existing config snapshot before ``before_attempt``."""
    limit = before_attempt if before_attempt is not None else 10_000
    for attempt in range(limit - 1, 0, -1):
        path = config_path_for(exp_id, attempt)
        if path.is_file():
            return path
    return None


def run_blocks_equal(left: dict, right: dict) -> bool:
    """Return whether two built configs use the same ``run`` block."""
    return left.get("run") == right.get("run")


def write_config(
    exp_id: str,
    *,
    attempt: int = 1,
    path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Build from registry and write a per-attempt config snapshot."""
    spec = get_experiment(exp_id)
    out = path or config_path_for(exp_id, attempt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_config(spec), indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def write_all_configs() -> list[pathlib.Path]:
    """Build current registry recipes (attempt-1 snapshots only)."""
    return [write_config(spec["id"], attempt=1) for spec in load_experiments()]
