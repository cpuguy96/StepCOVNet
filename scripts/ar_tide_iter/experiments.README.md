# experiments.json — registry vs retries

`experiments.json` holds **one entry per experiment id** (`iter31`, `iter32`, …). Each entry is the **current canonical recipe** for that id: `notes` plus `run` overrides merged onto `configs/ar/tide_overfit.json`.

It is **not** updated per retry attempt.

## What happens on each run

| Layer                                          | Role                                                        |
| ---------------------------------------------- | ----------------------------------------------------------- |
| `experiments.json`                             | Latest recipe for the id (edit when the hypothesis changes) |
| `logs/ar_tide_iter/configs/<id>.json`          | Frozen snapshot for attempt 1                               |
| `logs/ar_tide_iter/configs/<id>.attemptN.json` | Frozen snapshot for attempt N≥2                             |
| `logs/ar_tide_iter/results.jsonl`              | One row per attempt (`attempt`, `kind`, `recipe_changed`)   |

`run_exp.py` always trains/evals from a **config snapshot**, not from the registry directly.

## When to edit experiments.json

| Situation                                                           | Action                                                                                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **New hypothesis**                                                  | Add a **new id** (`iter39`, …) with a new `run` block                                                                                |
| **Same id, recipe fix**                                             | Edit the existing entry, then retry the same `--id` (attempt 2+ gets a new snapshot; log shows `recipe changed in experiments.json`) |
| **Same id, infra failure** (killed job, GPU busy, no recipe change) | **Do not** edit `experiments.json`; retry with `--reuse-last-config`                                                                 |

## Examples

Infra retry (same snapshot as last attempt):

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --id iter31 ^
    --reuse-last-config --retry-reason "killed mid-train"
```

Recipe change (edit `experiments.json` first):

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --id iter31 ^
    --retry-reason "bumped lambda_inc"
```

New experiment (preferred for a different hypothesis):

```text
# add iter39 to experiments.json, then:
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --id iter39
```

## Agent-planned runs (overnight)

The **agent** chooses the next recipe from training results — not a fixed queue and not auto-mutation code.

1. `venv\Scripts\python.exe scripts/ar_tide_iter/session_brief.py`
2. Agent writes `logs/ar_tide_iter/next_experiment.json` with **only the overrides** to apply (`run`, and optionally `model` / `dataset`). Diffs vs champion and prior runs are in the brief — there is no fixed knob list.
3. `venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --once`

`run_overnight.py` registers the plan into `experiments.json`, archives it under `logs/ar_tide_iter/applied_plans/`, and calls `run_exp.py`. Eval metrics and gates stay fixed across runs (see `recipe_diff.FIXED_EVAL_METRICS`). **`init_model_path` is stripped** — each run trains from scratch.
