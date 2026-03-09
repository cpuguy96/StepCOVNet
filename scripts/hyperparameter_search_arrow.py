r"""Hyperparameter search for the ARROW model.

Runs multiple training jobs from a sweep config. Supports grid search (all
combinations or first N) or random search (random subset of combinations).
Enforces val_take_count=-1 and batch_size=1. epoch and take_count can be set
in the base config or swept via search_space (run.epoch, run.take_count).
Records metrics per run and writes results plus best_config.json.

Training jobs can run in parallel with --workers. Each run writes to its own
subdirectory (run_0, run_1, ...) under the sweep output to avoid clashes.
When using multiple workers with a single GPU, consider --workers=1 or
setting CUDA_VISIBLE_DEVICES to avoid OOM or contention.

Usage:
    python scripts/hyperparameter_search_arrow.py --sweep_config=configs/arrow_sweep_example.json
    python scripts/hyperparameter_search_arrow.py --sweep_config=configs/arrow_sweep_example.json --max_runs=5
    python scripts/hyperparameter_search_arrow.py --sweep_config=configs/arrow_sweep_example.json --search=random --max_runs=10
    python scripts/hyperparameter_search_arrow.py --sweep_config=configs/arrow_sweep_example.json --search=random --max_runs=10 --seed=42
    python scripts/hyperparameter_search_arrow.py --sweep_config=configs/arrow_sweep_example.json --workers=2

Resume after a crash (run from project root, same sweep_output_dir as before):
    python scripts/hyperparameter_search_arrow.py --resume_from=output/hparam_search/arrow_sweep_20250225-123456
"""

from __future__ import annotations

import argparse
import datetime
import gc
import itertools
import json
import os
import random
import sys
from concurrent import futures
from typing import Any, cast

import tensorflow as tf

# Add project root for imports when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stepcovnet import config, trainers  # noqa: E402

PARSER = argparse.ArgumentParser(
    description="Run ARROW hyperparameter search (grid or random). epoch and take_count configurable via base config or search_space."
)
PARSER.add_argument(
    "--sweep_config",
    type=str,
    default=None,
    help="Path to sweep JSON (base_config, search_space, optimize). Required for a new sweep; do not set when using --resume_from.",
)
PARSER.add_argument(
    "--search",
    type=str,
    choices=("grid", "random"),
    default=None,
    help="Search strategy: grid or random. Default: from sweep config 'search', or 'grid' if unset.",
)
PARSER.add_argument(
    "--max_runs",
    type=int,
    default=None,
    help="Cap number of runs. For grid: take first max_runs; for random: sample this many (required for random if not in config).",
)
PARSER.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for random search (reproducibility). Overrides sweep config seed if set.",
)
PARSER.add_argument(
    "--workers",
    type=int,
    default=None,
    help="Number of training jobs to run in parallel. Overrides sweep config 'workers' when set; otherwise uses config or 1.",
)
PARSER.add_argument(
    "--resume_from",
    type=str,
    default=None,
    metavar="DIR",
    help="Resume a previous sweep from this output directory. Loads sweep_config.json and results.json, runs only missing runs. Cannot be used with --sweep_config.",
)


# Keys that must not be overridden by the search space (fixed or from base only).
_FORBIDDEN_OVERRIDE_KEYS = frozenset(
    {
        "dataset.batch_size",
        "dataset.snippet_half_frames",
        "model.snippet_half_frames",
        "run.val_take_count",
        "run.show_model_summary",
        "run.fit_verbose",
    }
)

# Fixed values applied to every run.
_VAL_TAKE_COUNT_FIXED = -1
_BATCH_SIZE_FIXED = 1


def load_sweep_config(path: str) -> dict[str, Any]:
    """Load and validate sweep config from JSON."""
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("sweep config must be a JSON object")
    if "base_config" not in data:
        raise ValueError("sweep config must contain 'base_config'")
    if "search_space" not in data:
        raise ValueError("sweep config must contain 'search_space'")
    if "optimize" not in data:
        raise ValueError("sweep config must contain 'optimize'")
    opt = data["optimize"]
    if not isinstance(opt, dict) or "metric" not in opt or "mode" not in opt:
        raise ValueError("sweep config 'optimize' must have 'metric' and 'mode'")
    if opt["mode"] not in ("min", "max"):
        raise ValueError("sweep config optimize.mode must be 'min' or 'max'")
    search_space = data["search_space"]
    if not isinstance(search_space, dict):
        raise ValueError("sweep config 'search_space' must be a JSON object")
    for key, values in search_space.items():
        if key in _FORBIDDEN_OVERRIDE_KEYS:
            raise ValueError(
                f"sweep config search_space must not contain forbidden key: {key!r}"
            )
        if not isinstance(values, list):
            raise ValueError(
                f"sweep config search_space[{key!r}] must be a list, got {type(values)}"
            )
    if "search" in data and data["search"] not in ("grid", "random"):
        raise ValueError(
            f"sweep config 'search' must be 'grid' or 'random', got {data['search']!r}"
        )
    if "workers" in data:
        w = data["workers"]
        if not isinstance(w, int) or w < 1:
            raise ValueError(
                f"sweep config 'workers' must be an integer >= 1, got {w!r}"
            )
    if "validity_gate" in data:
        vg = data["validity_gate"]
        if not isinstance(vg, dict):
            raise ValueError(
                "sweep config 'validity_gate' must be a JSON object when set"
            )
        if "min_fraction" not in vg:
            raise ValueError("sweep config 'validity_gate' must contain 'min_fraction'")
        mf = vg["min_fraction"]
        if not isinstance(mf, (int, float)) or not (0.0 <= mf <= 1.0):
            raise ValueError(
                "sweep config validity_gate.min_fraction must be a number in [0, 1], "
                f"got {mf!r}"
            )
        if "validity_metric" in vg and vg["validity_metric"] is not None:
            if not isinstance(vg["validity_metric"], str):
                raise ValueError(
                    "sweep config validity_gate.validity_metric must be a string "
                    f"when set, got {type(vg['validity_metric']).__name__}"
                )
            if not vg["validity_metric"].strip():
                raise ValueError(
                    "sweep config validity_gate.validity_metric must be a non-empty "
                    "string when set"
                )
        if "optimize_metric" in vg and vg["optimize_metric"] is not None:
            if not isinstance(vg["optimize_metric"], str):
                raise ValueError(
                    "sweep config validity_gate.optimize_metric must be a string "
                    f"when set, got {type(vg['optimize_metric']).__name__}"
                )
        if "optimize_mode" in vg and vg["optimize_mode"] is not None:
            if vg["optimize_mode"] not in ("min", "max"):
                raise ValueError(
                    f"sweep config validity_gate.optimize_mode must be 'min' or 'max', "
                    f"got {vg['optimize_mode']!r}"
                )
    return data


def _load_resume_state(
    sweep_output_dir: str,
) -> tuple[dict[str, Any], list[dict[str, Any] | None]]:
    """Load sweep config and partial/full results from a previous run for resume.

    Returns (sweep_save, results_list). results_list has length = total runs;
    completed runs have a dict, incomplete have None.
    """
    config_path = os.path.join(sweep_output_dir, "sweep_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Cannot resume: {config_path!r} not found. Use the exact sweep output directory from the interrupted run."
        )
    with open(config_path) as f:
        sweep_save = json.load(f)
    results_path = os.path.join(sweep_output_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results_list = json.load(f)
        if not isinstance(results_list, list):
            raise ValueError(
                f"Resume results.json must be a list, got {type(results_list).__name__}"
            )
    else:
        results_list = []
    return sweep_save, results_list


def _rebuild_combinations_from_saved(
    sweep_save: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the same combination list from a saved sweep_config.json."""
    search_space = sweep_save["search_space"]
    full_combinations = expand_grid(search_space)
    effective_search = sweep_save.get("_effective_search", "grid")
    effective_seed = sweep_save.get("_effective_seed")
    max_runs = sweep_save.get("max_runs")
    if effective_search == "random":
        if effective_seed is not None:
            random.seed(effective_seed)
        n_sample = min(max_runs or len(full_combinations), len(full_combinations))
        return random.sample(full_combinations, n_sample)
    return full_combinations[:max_runs] if max_runs is not None else full_combinations


def _expand_grid_simple(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of all keys in search_space."""
    keys = list(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    combinations = []
    for combo in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, combo, strict=False)))
    return combinations


def expand_grid(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Build all combinations from search_space.

    When search_space contains "model.model_type" with multiple values, expands
    per model type: each combination includes only model.model_type and
    model.<that_type>.* overrides (plus dataset.*, run.*). This yields valid
    combinations when sweeping over architectures without creating a full
    Cartesian product that filter_valid_model_combinations would drop.
    """
    model_type_key = "model.model_type"
    if model_type_key not in search_space:
        return _expand_grid_simple(search_space)
    model_type_values = search_space[model_type_key]
    if not isinstance(model_type_values, list) or len(model_type_values) <= 1:
        return _expand_grid_simple(search_space)
    result = []
    for mt in model_type_values:
        sub_space = {}
        for k, v in search_space.items():
            if k == model_type_key:
                sub_space[k] = [mt]
            elif k.startswith("model.") and k != model_type_key:
                parts = k.split(".")
                if len(parts) >= 2 and parts[1] == mt:
                    sub_space[k] = v
            else:
                sub_space[k] = v
        result.extend(_expand_grid_simple(sub_space))
    return result


def filter_valid_model_combinations(
    combinations: list[dict[str, Any]],
    default_model_type: str,
) -> list[dict[str, Any]]:
    """Keep only combinations where model.<block>.<param> keys match effective model_type.

    For each combination, effective model_type is overrides.get("model.model_type")
    or default_model_type. A combination is kept iff every key of the form
    model.<block>.<param> has block == effective model_type. Non-model-block keys
    (dataset.*, run.*, model.model_type) are always allowed.

    Args:
        combinations: List of override dicts from expand_grid.
        default_model_type: model_type from base config when not overridden.

    Returns:
        Filtered list of combinations (only valid model-specific overrides).
    """
    result = []
    for combo in combinations:
        effective = combo.get("model.model_type", default_model_type)
        valid = True
        for key in combo:
            if not key.startswith("model.") or key == "model.model_type":
                continue
            parts = key.split(".")
            if len(parts) >= 3 and parts[1] != effective:
                valid = False
                break
        if valid:
            result.append(combo)
    return result


def _set_nested(d: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a possibly nested key (e.g. 'transformer.num_layers') in d, creating dicts as needed.

    Missing or None intermediate segments are created as empty dicts so nested keys can be set.
    Raises ValueError if an intermediate segment exists and is a non-dict, non-None value (e.g.
    setting transformer.num_layers when transformer is already a scalar).
    """
    parts = key_path.split(".")
    current = d
    for i, part in enumerate(parts[:-1]):
        existing = current.get(part)
        if part not in current or existing is None:
            current[part] = {}
        elif not isinstance(existing, dict):
            segment = ".".join(parts[: i + 1])
            raise ValueError(
                f"Cannot set nested key '{key_path}': segment '{segment}' is not a "
                "nested object (leaf value); use a path without extra segments."
            )
        current = current[part]
    current[parts[-1]] = value


def apply_overrides(
    base: config.ArrowExperimentConfig,
    overrides: dict[str, Any],
) -> config.ArrowExperimentConfig:
    """Apply overrides to base config and enforce fixed values.

    Does not apply forbidden keys. After overrides, forces run.val_take_count=-1,
    dataset.batch_size=1, and sweep-time verbosity (show_model_summary=False,
    fit_verbose=0). run.epoch and run.take_count come from base config or overrides.
    Model overrides use dotted paths: model.model_type or model.<block>.<param>
    (e.g. model.transformer.num_layers, model.mlp.hidden_dims).
    """
    # Work with dict so we can mutate nested fields.
    d = base.as_dict()
    for key, value in overrides.items():
        if key in _FORBIDDEN_OVERRIDE_KEYS:
            continue
        if "." not in key:
            continue
        prefix, rest = key.split(".", 1)
        if prefix == "dataset":
            _set_nested(d["dataset"], rest, value)
        elif prefix == "model":
            if rest != "model_type" and "." not in rest:
                raise ValueError(
                    f"Model override key must be model.model_type or model.<block>.<param>, got model.{rest}"
                )
            _set_nested(d["model"], rest, value)
        elif prefix == "run":
            _set_nested(d["run"], rest, value)
    # Force fixed values
    d["run"]["val_take_count"] = _VAL_TAKE_COUNT_FIXED
    d["dataset"]["batch_size"] = _BATCH_SIZE_FIXED
    d["run"]["show_model_summary"] = False
    d["run"]["fit_verbose"] = 0
    return config.ArrowExperimentConfig.from_dict(d)


def _clear_tf_memory() -> None:
    """Clear TensorFlow/Keras session and run GC to free memory between runs.

    TF keeps graph and allocator state across model builds; without this, each
    training run in the same process adds memory (in-process sweep or
    reused multiprocessing workers).
    """
    try:
        backend = getattr(getattr(tf, "keras", None), "backend", None)
        if backend is not None:
            backend.clear_session(free_memory=True)
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    gc.collect()


def extract_metrics(history: Any) -> dict[str, Any]:
    """Extract best and final metrics from training history."""
    h = history.history
    metric_names = [k for k in h if k.startswith("val_")]
    result = {}
    for name in metric_names:
        vals = h[name]
        if not vals:
            continue
        result[f"final_{name}"] = float(vals[-1])
        # Loss metrics (val_loss, val_main_loss, val_*_aux_loss, etc.) are minimized.
        is_loss = name.endswith("_loss")
        best_val = min(vals) if is_loss else max(vals)
        result[f"best_{name}"] = float(best_val)
        best_epoch_1based = int(vals.index(best_val) + 1)
        result[f"best_epoch_{name}"] = best_epoch_1based
    return result


def _run_single_training(
    run_index: int,
    overrides: dict[str, Any],
    base_config_dict: dict[str, Any],
    sweep_output_dir: str,
    project_root: str,
    effective_seed: int | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Run one training job (for use in a worker process).

    Ensures project root is on sys.path, builds config, sets per-run output
    dirs, runs training, returns (run_index, metrics, overrides).
    If effective_seed is set (from sweep config or --seed), it is applied to
    run_config.run.seed so training uses the same seed for reproducibility.
    """
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    base = config.ArrowExperimentConfig.from_dict(base_config_dict)
    run_config = apply_overrides(base, overrides)
    if effective_seed is not None:
        run_config.run.seed = effective_seed
    run_config.run.model_output_dir = os.path.join(
        sweep_output_dir, "models", f"run_{run_index}"
    )
    run_config.run.callback_root_dir = os.path.join(
        sweep_output_dir, "callbacks", f"run_{run_index}"
    )
    os.makedirs(run_config.run.model_output_dir, exist_ok=True)
    os.makedirs(run_config.run.callback_root_dir, exist_ok=True)
    _model, history = trainers.run_arrow_train_from_config(run_config)
    metrics = extract_metrics(history)
    # Clear TF session so this worker doesn't keep memory when reused for next run
    del _model, history
    _clear_tf_memory()
    return (run_index, metrics, overrides)


def select_best_run(
    results: list[dict[str, Any]],
    metric: str,
    mode: str,
) -> int:
    """Return index of the best run (0-based). metric e.g. 'val_loss', mode 'min' or 'max'."""
    best_key = f"best_{metric}"
    if not results:
        raise ValueError("results list is empty")
    for r in results:
        if best_key not in r:
            raise ValueError(
                f"metric {best_key!r} not found in result keys: {list(r.keys())}"
            )
    if mode == "min":
        return int(min(range(len(results)), key=lambda i: results[i][best_key]))
    return int(max(range(len(results)), key=lambda i: results[i][best_key]))


def _resolve_validity_metric(
    results: list[dict[str, Any]],
    explicit_metric: str | None,
) -> str | None:
    """Return the best_* key used for validity gating, or None if not found.

    If explicit_metric is set (e.g. 'val_chart_validity_pass_rate_0_99'), return
    'best_' + explicit_metric. Otherwise auto-detect from the first result's keys:
    prefer any key matching best_val_chart_validity_pass_rate_*, else
    best_val_chart_validity.
    """
    if explicit_metric is not None:
        return f"best_{explicit_metric}"
    if not results:
        return None
    keys = list(results[0].keys())
    pass_rate_keys = [
        k for k in keys if k.startswith("best_val_chart_validity_pass_rate_")
    ]
    if pass_rate_keys:
        return min(pass_rate_keys)
    if "best_val_chart_validity" in keys:
        return "best_val_chart_validity"
    return None


def _select_best_run_with_validity_gate(
    results: list[dict[str, Any]],
    sweep_save: dict[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    """Return (best_run_index, gate_info).

    If validity_gate is not in sweep_save, returns select_best_run by main optimize
    metric and gate_info=None. Otherwise filters to runs with validity >= min_fraction,
    selects best among them by gate's optimize_metric/mode (default val_arrow_dist_match
    max). If no run passes the gate, falls back to main optimize and gate_info includes
    used_fallback=True.
    """
    optimize = sweep_save["optimize"]
    main_metric = optimize["metric"]
    main_mode = optimize["mode"]
    vg = sweep_save.get("validity_gate")
    if not vg:
        idx = select_best_run(results, main_metric, main_mode)
        return (idx, None)
    min_fraction = float(vg["min_fraction"])
    validity_metric = vg.get("validity_metric")
    best_validity_key = _resolve_validity_metric(results, validity_metric)
    if best_validity_key is None:
        idx = select_best_run(results, main_metric, main_mode)
        print(
            "  WARNING: validity_gate set but no validity metric found in results; "
            "selecting by main optimize metric."
        )
        return (
            idx,
            {
                "validity_metric": validity_metric or "(auto)",
                "min_fraction": min_fraction,
                "n_passed": 0,
                "n_total": len(results),
                "used_fallback": True,
            },
        )
    valid_indices = [
        i
        for i in range(len(results))
        if results[i].get(best_validity_key, 0.0) >= min_fraction
    ]
    gate_metric = vg.get("optimize_metric") or "val_arrow_dist_match"
    gate_mode = vg.get("optimize_mode") or "max"
    validity_metric_name = validity_metric or best_validity_key.replace("best_", "", 1)
    gate_info = {
        "validity_metric": validity_metric_name,
        "min_fraction": min_fraction,
        "n_passed": len(valid_indices),
        "n_total": len(results),
        "used_fallback": False,
        "optimize_metric": gate_metric,
        "optimize_mode": gate_mode,
    }
    if valid_indices:
        valid_results = [results[i] for i in valid_indices]
        best_in_valid = select_best_run(valid_results, gate_metric, gate_mode)
        return (valid_indices[best_in_valid], gate_info)
    idx = select_best_run(results, main_metric, main_mode)
    print(
        "  WARNING: No run met validity gate "
        f"({best_validity_key} >= {min_fraction}); selecting by main optimize metric."
    )
    gate_info["used_fallback"] = True
    return (idx, gate_info)


def _format_overrides(overrides: dict[str, Any]) -> str:
    """Format overrides as aligned key: value lines."""
    if not overrides:
        return "  (none)"
    lines = []
    for k, v in sorted(overrides.items()):
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _print_sweep_header(
    total_runs: int,
    sweep_output_dir: str,
    optimize_metric: str,
    optimize_mode: str,
    base_config_path: str,
    search: str = "grid",
    full_grid_size: int | None = None,
) -> None:
    """Print a clear header when the sweep starts."""
    width = 60
    print()
    print("=" * width)
    print("  ARROW Hyperparameter Sweep")
    print("=" * width)
    print(f"  Base config:    {base_config_path}")
    print(f"  Output dir:    {sweep_output_dir}")
    print(f"  Search:        {search}")
    if (
        search == "random"
        and full_grid_size is not None
        and full_grid_size != total_runs
    ):
        print(
            f"  Total runs:    {total_runs} (sampled from {full_grid_size} combinations)"
        )
    else:
        print(f"  Total runs:    {total_runs}")
    print(f"  Optimize:      {optimize_mode} {optimize_metric}")
    print("=" * width)
    print()


def _print_run_header(
    run_index: int, total_runs: int, overrides: dict[str, Any]
) -> None:
    """Print a per-run banner with overrides."""
    width = 60
    title = f" Run {run_index + 1}/{total_runs} "
    padding = (width - len(title)) // 2
    print()
    print("-" * width)
    print(" " * padding + title + " " * (width - padding - len(title)))
    print("-" * width)
    print(_format_overrides(overrides))
    print("-" * width)


def _print_run_result(metrics: dict[str, Any], optimize_metric: str) -> None:
    """Print key metric after a run completes."""
    best_key = f"best_{optimize_metric}"
    val = metrics.get(best_key)
    if val is not None:
        print(f"  -> {best_key}: {val:.6f}")
    print()


def _is_better_than(current_val: float, best_so_far: float | None, mode: str) -> bool:
    """Return True if current_val is better than best_so_far (or best_so_far is None)."""
    if best_so_far is None:
        return True
    if mode == "min":
        return current_val < best_so_far
    return current_val > best_so_far


def _print_best_so_far(
    run_index: int,
    total_runs: int,
    best_key: str,
    value: float,
    overrides: dict[str, Any],
) -> None:
    """Print message when this run has the best val metrics seen so far."""
    print(
        f"  *** NEW BEST (run {run_index + 1}/{total_runs}): {best_key} = {value:.6f} ***"
    )
    print(_format_overrides(overrides))
    print()


def _print_sweep_summary(
    results: list[dict[str, Any]],
    best_idx: int,
    best_overrides: dict[str, Any],
    optimize_metric: str,
    optimize_mode: str,
    results_path: str,
    best_config_path: str,
    gate_info: dict[str, Any] | None = None,
) -> None:
    """Print final summary with best run and file paths."""
    width = 60
    if gate_info and not gate_info.get("used_fallback"):
        display_metric = gate_info["optimize_metric"]
        display_mode = gate_info["optimize_mode"]
    else:
        display_metric = optimize_metric
        display_mode = optimize_mode
    best_key = f"best_{display_metric}"
    best_val = results[best_idx][best_key]

    print()
    print("=" * width)
    print("  SWEEP COMPLETE")
    print("=" * width)
    print()
    if gate_info:
        print(
            f"  Validity gate: {gate_info['n_passed']} of {gate_info['n_total']} runs "
            f"passed ({gate_info['validity_metric']} >= {gate_info['min_fraction']})"
        )
        if gate_info.get("used_fallback"):
            print("  (No run met gate; best run selected by main optimize metric.)")
        print()
    print("  Best run:")
    print(f"    Index:    {best_idx + 1} (of {len(results)})")
    print(f"    Optimize: {display_mode} {display_metric}")
    print(f"    {best_key}: {best_val:.6f}")
    print("    Overrides:")
    for k, v in sorted(best_overrides.items()):
        print(f"      {k}: {v}")
    print()
    print("  Output files:")
    print(f"    Results:     {results_path}")
    print(f"    Best config: {best_config_path}")
    print()
    print("=" * width)


class _SweepContext:
    """State for a sweep (fresh or resumed): config, combinations, output dir, and results.

    Plain class (no dataclass) so the script can be loaded via importlib in tests.

    Attributes:
        base_config: Loaded ArrowExperimentConfig used as the base for overrides.
        base_path: Path to the base config file.
        combinations: List of override dicts (one per run).
        sweep_output_dir: Directory where sweep results are written.
        sweep_save: Dict of sweep metadata for resume (saved to disk).
        effective_search: Search strategy in use (e.g. 'grid', 'random').
        effective_seed: Random seed for reproducible sweeps (or None).
        full_combinations: Full list of all combination dicts.
        results_by_index: Map from run index to result dict (or None if pending).
        pending: List of (index, overrides) for runs not yet completed.
    """

    __slots__ = (
        "base_config",
        "base_path",
        "combinations",
        "sweep_output_dir",
        "sweep_save",
        "effective_search",
        "effective_seed",
        "full_combinations",
        "results_by_index",
        "pending",
    )

    def __init__(
        self,
        *,
        base_config: Any,
        base_path: str,
        combinations: list[dict[str, Any]],
        sweep_output_dir: str,
        sweep_save: dict[str, Any],
        effective_search: str,
        effective_seed: int | None,
        full_combinations: list[dict[str, Any]],
        results_by_index: dict[int, dict[str, Any]],
        pending: list[tuple[int, dict[str, Any]]],
    ) -> None:
        self.base_config = base_config
        self.base_path = base_path
        self.combinations = combinations
        self.sweep_output_dir = sweep_output_dir
        self.sweep_save = sweep_save
        self.effective_search = effective_search
        self.effective_seed = effective_seed
        self.full_combinations = full_combinations
        self.results_by_index = results_by_index
        self.pending = pending


def _setup_resume(resume_from: str) -> _SweepContext | None:
    """Load state from a previous run. If all runs are already complete, write best_config and return None."""
    if not os.path.isabs(resume_from):
        resume_from = os.path.join(_PROJECT_ROOT, resume_from)
    if not os.path.isdir(resume_from):
        PARSER.error(f"--resume_from directory does not exist: {resume_from}")
    sweep_save, results_list = _load_resume_state(resume_from)
    base_path = sweep_save["base_config"]
    if not os.path.isabs(base_path):
        base_path = os.path.join(_PROJECT_ROOT, base_path)
    base_config = config.ArrowExperimentConfig.from_json(base_path)
    combinations = _rebuild_combinations_from_saved(sweep_save)
    if len(results_list) < len(combinations):
        results_list.extend([None] * (len(combinations) - len(results_list)))
    else:
        results_list = results_list[: len(combinations)]
    completed = {i for i in range(len(combinations)) if results_list[i] is not None}
    pending = [
        (i, overrides) for i, overrides in enumerate(combinations) if i not in completed
    ]
    results_by_index = cast(
        dict[int, dict[str, Any]],
        {
            i: results_list[i]
            for i in range(len(combinations))
            if results_list[i] is not None
        },
    )
    if not pending:
        _finish_sweep_early(
            sweep_output_dir=resume_from,
            base_config=base_config,
            combinations=combinations,
            results_by_index=results_by_index,
            sweep_save=sweep_save,
        )
        return None
    print(f"Resume: {len(completed)} runs already done, {len(pending)} remaining.\n")
    return _SweepContext(
        base_config=base_config,
        base_path=base_path,
        combinations=combinations,
        sweep_output_dir=resume_from,
        sweep_save=sweep_save,
        effective_search=sweep_save.get("_effective_search", "grid"),
        effective_seed=sweep_save.get("_effective_seed"),
        full_combinations=expand_grid(sweep_save["search_space"]),
        results_by_index=results_by_index,
        pending=pending,
    )


def _finish_sweep_early(
    sweep_output_dir: str,
    base_config: config.ArrowExperimentConfig,
    combinations: list[dict[str, Any]],
    results_by_index: dict[int, dict[str, Any]],
    sweep_save: dict[str, Any],
) -> None:
    """Write best_config and summary when resuming with no pending runs."""
    optimize_metric = sweep_save["optimize"]["metric"]
    optimize_mode = sweep_save["optimize"]["mode"]
    results: list[dict[str, Any]] = [
        results_by_index[i] for i in range(len(combinations))
    ]
    best_idx, gate_info = _select_best_run_with_validity_gate(results, sweep_save)
    best_overrides = results[best_idx]["overrides"]
    best_config = apply_overrides(base_config, best_overrides)
    callback_root_dir = os.path.join(sweep_output_dir, "callbacks")
    model_output_dir = os.path.join(sweep_output_dir, "models")
    best_config.run.model_output_dir = os.path.join(model_output_dir, f"run_{best_idx}")
    best_config.run.callback_root_dir = os.path.join(
        callback_root_dir, f"run_{best_idx}"
    )
    best_config.run.val_take_count = _VAL_TAKE_COUNT_FIXED
    best_config.dataset.batch_size = _BATCH_SIZE_FIXED
    best_config_path = os.path.join(sweep_output_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config.as_dict(), f, indent=2)
    results_path = os.path.join(sweep_output_dir, "results.json")
    _print_sweep_summary(
        results=results,
        best_idx=best_idx,
        best_overrides=best_overrides,
        optimize_metric=optimize_metric,
        optimize_mode=optimize_mode,
        results_path=results_path,
        best_config_path=best_config_path,
        gate_info=gate_info,
    )


def _setup_fresh_sweep(args: argparse.Namespace) -> _SweepContext:
    """Load sweep config, build combinations, create output dir, save sweep_config.json."""
    sweep = load_sweep_config(args.sweep_config)
    base_path = sweep["base_config"]
    if not os.path.isabs(base_path):
        base_path = os.path.join(_PROJECT_ROOT, base_path)
    base_config = config.ArrowExperimentConfig.from_json(base_path)
    effective_search = (
        args.search if args.search is not None else sweep.get("search") or "grid"
    )
    if effective_search not in ("grid", "random"):
        PARSER.error(f"--search must be 'grid' or 'random', got {effective_search!r}")
    max_runs = args.max_runs if args.max_runs is not None else sweep.get("max_runs")
    if max_runs is not None and max_runs <= 0:
        PARSER.error(f"max_runs must be > 0 when set (got {max_runs})")
    search_space = sweep["search_space"]
    full_combinations = expand_grid(search_space)
    raw_count = len(full_combinations)
    full_combinations = filter_valid_model_combinations(
        full_combinations, base_config.model.model_type
    )
    excluded = raw_count - len(full_combinations)
    if excluded > 0:
        print(
            f"Filtered to {len(full_combinations)} valid model-specific combinations "
            f"({excluded} excluded)."
        )
    effective_seed = args.seed if args.seed is not None else sweep.get("seed")
    if effective_search == "random":
        if effective_seed is not None:
            random.seed(effective_seed)
        n_sample = min(max_runs or len(full_combinations), len(full_combinations))
        combinations = random.sample(full_combinations, n_sample)
    else:
        combinations = (
            full_combinations[:max_runs] if max_runs is not None else full_combinations
        )
    sweep_output_dir = sweep.get("sweep_output_dir")
    if not sweep_output_dir:
        sweep_output_dir = os.path.join(
            _PROJECT_ROOT,
            "output",
            f"arrow_sweep_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
    else:
        if not os.path.isabs(sweep_output_dir):
            sweep_output_dir = os.path.join(
                _PROJECT_ROOT,
                sweep_output_dir,
                f"arrow_sweep_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )
    os.makedirs(sweep_output_dir, exist_ok=True)
    callback_root_dir = os.path.join(sweep_output_dir, "callbacks")
    model_output_dir = os.path.join(sweep_output_dir, "models")
    os.makedirs(callback_root_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)
    sweep_save = {**sweep, "base_config": base_path}
    sweep_save["_effective_search"] = effective_search
    if effective_seed is not None:
        sweep_save["_effective_seed"] = effective_seed
    with open(os.path.join(sweep_output_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_save, f, indent=2)
    return _SweepContext(
        base_config=base_config,
        base_path=base_path,
        combinations=combinations,
        sweep_output_dir=sweep_output_dir,
        sweep_save=sweep_save,
        effective_search=effective_search,
        effective_seed=effective_seed,
        full_combinations=full_combinations,
        results_by_index={},
        pending=list(enumerate(combinations)),
    )


def _run_pending(ctx: _SweepContext, workers: int) -> None:
    """Run pending training jobs, update ctx.results_by_index, write checkpoints, print progress."""
    optimize_metric = ctx.sweep_save["optimize"]["metric"]
    optimize_mode = ctx.sweep_save["optimize"]["mode"]
    best_key = f"best_{optimize_metric}"
    best_metric_so_far: float | None = None
    for _idx, r in ctx.results_by_index.items():
        val = r.get(best_key)
        if val is not None and _is_better_than(val, best_metric_so_far, optimize_mode):
            best_metric_so_far = val
    _print_sweep_header(
        total_runs=len(ctx.combinations),
        sweep_output_dir=ctx.sweep_output_dir,
        optimize_metric=optimize_metric,
        optimize_mode=optimize_mode,
        base_config_path=ctx.base_path,
        search=ctx.effective_search,
        full_grid_size=(
            len(ctx.full_combinations) if ctx.effective_search == "random" else None
        ),
    )
    if workers > 1:
        print(f"  Workers:       {workers} (parallel)\n")
    results_path = os.path.join(ctx.sweep_output_dir, "results.json")

    def write_checkpoint() -> None:
        ordered = [ctx.results_by_index.get(i) for i in range(len(ctx.combinations))]
        with open(results_path, "w") as f:
            json.dump(ordered, f, indent=2)

    base_config_dict = ctx.base_config.as_dict()
    with futures.ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=1,
    ) as executor:
        future_to_index = {
            executor.submit(
                _run_single_training,
                i,
                overrides,
                base_config_dict,
                ctx.sweep_output_dir,
                _PROJECT_ROOT,
                ctx.effective_seed,
            ): i
            for i, overrides in ctx.pending
        }
        for future in futures.as_completed(future_to_index):
            run_index = future_to_index[future]
            run_index_result, metrics, overrides_result = future.result()
            ctx.results_by_index[run_index] = {
                "run_index": run_index_result,
                "overrides": overrides_result,
                **metrics,
            }
            write_checkpoint()
            _print_run_header(run_index, len(ctx.combinations), overrides_result)
            _print_run_result(metrics, optimize_metric)
            val = metrics.get(best_key)
            if val is not None and _is_better_than(
                val, best_metric_so_far, optimize_mode
            ):
                best_metric_so_far = val
                _print_best_so_far(
                    run_index, len(ctx.combinations), best_key, val, overrides_result
                )


def _write_final_results_and_best(ctx: _SweepContext) -> None:
    """Build full results list, write results.json and best_config.json, print summary."""
    results: list[dict[str, Any]] = [
        ctx.results_by_index[i] for i in range(len(ctx.combinations))
    ]
    results_path = os.path.join(ctx.sweep_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    optimize_metric = ctx.sweep_save["optimize"]["metric"]
    optimize_mode = ctx.sweep_save["optimize"]["mode"]
    best_idx, gate_info = _select_best_run_with_validity_gate(results, ctx.sweep_save)
    best_overrides = results[best_idx]["overrides"]
    best_config = apply_overrides(ctx.base_config, best_overrides)
    callback_root_dir = os.path.join(ctx.sweep_output_dir, "callbacks")
    model_output_dir = os.path.join(ctx.sweep_output_dir, "models")
    best_config.run.model_output_dir = os.path.join(model_output_dir, f"run_{best_idx}")
    best_config.run.callback_root_dir = os.path.join(
        callback_root_dir, f"run_{best_idx}"
    )
    best_config.run.val_take_count = _VAL_TAKE_COUNT_FIXED
    best_config.dataset.batch_size = _BATCH_SIZE_FIXED
    best_config_path = os.path.join(ctx.sweep_output_dir, "best_config.json")
    with open(best_config_path, "w") as f:
        json.dump(best_config.as_dict(), f, indent=2)
    _print_sweep_summary(
        results=results,
        best_idx=best_idx,
        best_overrides=best_overrides,
        optimize_metric=optimize_metric,
        optimize_mode=optimize_mode,
        results_path=results_path,
        best_config_path=best_config_path,
        gate_info=gate_info,
    )


def main() -> int:
    args = PARSER.parse_args()
    if not args.resume_from and not args.sweep_config:
        PARSER.error("--sweep_config is required unless --resume_from is set")
    if args.resume_from and args.sweep_config:
        PARSER.error(
            "Cannot set both --resume_from and --sweep_config; use only --resume_from to resume."
        )

    if args.resume_from:
        ctx = _setup_resume(args.resume_from)
        if ctx is None:
            return 0
    else:
        ctx = _setup_fresh_sweep(args)

    workers = (
        args.workers if args.workers is not None else ctx.sweep_save.get("workers", 1)
    )
    if not isinstance(workers, int) or workers < 1:
        PARSER.error(
            "--workers / sweep config 'workers' must be an integer >= 1, got "
            f"{workers!r}"
        )
    _run_pending(ctx, workers)
    _write_final_results_and_best(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
