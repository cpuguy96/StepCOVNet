r"""Hyperparameter search for the ARROW model.

Runs multiple training jobs from a sweep config (grid search). Enforces
val_take_count=-1 and batch_size=1. epoch and take_count can be set in the
base config or swept via search_space (run.epoch, run.take_count). Records
metrics per run and writes results plus best_config.json.

Usage:
    python scripts/hyperparameter_search_arrow.py --sweep_config=configs/arrow_sweep_example.json
    python scripts/hyperparameter_search_arrow.py --sweep_config=configs/arrow_sweep_example.json --max_runs=5
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from typing import Any

# Add project root for imports when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stepcovnet import config
from stepcovnet import trainers

PARSER = argparse.ArgumentParser(
    description="Run ARROW hyperparameter search (grid). epoch and take_count configurable via base config or search_space."
)
PARSER.add_argument(
    "--sweep_config",
    type=str,
    required=True,
    help="Path to sweep JSON (base_config, search_space, optimize).",
)
PARSER.add_argument(
    "--max_runs",
    type=int,
    default=None,
    help="Cap number of runs. Overrides sweep config max_runs if set (default: use config or no cap).",
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
    with open(path, "r") as f:
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
    return data


def expand_grid(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Build all combinations from search_space (Cartesian product)."""
    keys = list(search_space.keys())
    value_lists = [search_space[k] for k in keys]
    combinations = []
    for combo in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def apply_overrides(
    base: config.ArrowExperimentConfig,
    overrides: dict[str, Any],
) -> config.ArrowExperimentConfig:
    """Apply overrides to base config and enforce fixed values.

    Does not apply forbidden keys. After overrides, forces run.val_take_count=-1,
    dataset.batch_size=1, and sweep-time verbosity (show_model_summary=False,
    fit_verbose=0). run.epoch and run.take_count come from base config or overrides.
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
            d["dataset"][rest] = value
        elif prefix == "model":
            d["model"][rest] = value
        elif prefix == "run":
            d["run"][rest] = value
    # Force fixed values
    d["run"]["val_take_count"] = _VAL_TAKE_COUNT_FIXED
    d["dataset"]["batch_size"] = _BATCH_SIZE_FIXED
    d["run"]["show_model_summary"] = False
    d["run"]["fit_verbose"] = 0
    return config.ArrowExperimentConfig.from_dict(d)


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
        is_loss = name == "val_loss"
        best_val = min(vals) if is_loss else max(vals)
        result[f"best_{name}"] = float(best_val)
        best_epoch_1based = int((vals.index(best_val) + 1))
        result[f"best_epoch_{name}"] = best_epoch_1based
    return result


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
) -> None:
    """Print a clear header when the sweep starts."""
    width = 60
    print()
    print("=" * width)
    print("  ARROW Hyperparameter Sweep")
    print("=" * width)
    print(f"  Base config:    {base_config_path}")
    print(f"  Output dir:    {sweep_output_dir}")
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


def _print_run_result(
    run_index: int, metrics: dict[str, Any], optimize_metric: str
) -> None:
    """Print key metric after a run completes."""
    best_key = f"best_{optimize_metric}"
    val = metrics.get(best_key)
    if val is not None:
        print(f"  -> {best_key}: {val:.6f}")
    print()


def _print_sweep_summary(
    results: list[dict[str, Any]],
    best_idx: int,
    best_overrides: dict[str, Any],
    optimize_metric: str,
    optimize_mode: str,
    results_path: str,
    best_config_path: str,
) -> None:
    """Print final summary with best run and file paths."""
    width = 60
    best_key = f"best_{optimize_metric}"
    best_val = results[best_idx][best_key]

    print()
    print("=" * width)
    print("  SWEEP COMPLETE")
    print("=" * width)
    print()
    print("  Best run:")
    print(f"    Index:   {best_idx + 1} (of {len(results)})")
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


def main() -> int:
    args = PARSER.parse_args()
    sweep = load_sweep_config(args.sweep_config)
    base_path = sweep["base_config"]
    if not os.path.isabs(base_path):
        base_path = os.path.join(_PROJECT_ROOT, base_path)
    base_config = config.ArrowExperimentConfig.from_json(base_path)

    search_space = sweep["search_space"]
    combinations = expand_grid(search_space)
    max_runs = args.max_runs if args.max_runs is not None else sweep.get("max_runs")
    if max_runs is not None:
        combinations = combinations[:max_runs]

    sweep_output_dir = sweep.get("sweep_output_dir")
    if not sweep_output_dir:
        sweep_output_dir = os.path.join(
            _PROJECT_ROOT,
            "callbacks",
            f"arrow_sweep_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
    else:
        if not os.path.isabs(sweep_output_dir):
            sweep_output_dir = os.path.join(_PROJECT_ROOT, sweep_output_dir)
    os.makedirs(sweep_output_dir, exist_ok=True)

    callback_root_dir = os.path.join(sweep_output_dir, "callbacks")
    model_output_dir = os.path.join(sweep_output_dir, "models")
    os.makedirs(callback_root_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)

    # Save sweep config for reproducibility
    sweep_save = {**sweep, "base_config": base_path}
    with open(os.path.join(sweep_output_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_save, f, indent=2)

    optimize_metric = sweep["optimize"]["metric"]
    optimize_mode = sweep["optimize"]["mode"]

    _print_sweep_header(
        total_runs=len(combinations),
        sweep_output_dir=sweep_output_dir,
        optimize_metric=optimize_metric,
        optimize_mode=optimize_mode,
        base_config_path=base_path,
    )

    results = []
    for i, overrides in enumerate(combinations):
        run_config = apply_overrides(base_config, overrides)
        run_config.run.model_output_dir = model_output_dir
        run_config.run.callback_root_dir = callback_root_dir

        _print_run_header(i, len(combinations), overrides)
        model, history = trainers.run_arrow_train_from_config(
            run_config.dataset,
            run_config.model,
            run_config.run,
        )
        metrics = extract_metrics(history)
        row = {"run_index": i, "overrides": overrides, **metrics}
        results.append(row)
        _print_run_result(i, metrics, optimize_metric)

    # Write results
    results_path = os.path.join(sweep_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    best_idx = select_best_run(results, optimize_metric, optimize_mode)
    best_overrides = results[best_idx]["overrides"]
    best_config = apply_overrides(base_config, best_overrides)
    best_config.run.model_output_dir = model_output_dir
    best_config.run.callback_root_dir = callback_root_dir
    best_config.run.val_take_count = _VAL_TAKE_COUNT_FIXED
    best_config.dataset.batch_size = _BATCH_SIZE_FIXED

    best_config_path = os.path.join(sweep_output_dir, "best_config.json")
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
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
