# Profile: ar-tide-overfit

Single-chart **tide** AR overfit — scratch train, teacher gate, then free-run **634/634 @ 20 ms**.

Handoff: [ar-tide-overnight-prompt.md](../../../../docs/agents/ar-tide-overnight-prompt.md).  
Metrics: [ONSET_METRICS.md](../../../../docs/research/ONSET_METRICS.md).  
Design: [AR_ONSET_DESIGN.md](../../../../docs/research/AR_ONSET_DESIGN.md).

## Success criteria

| Phase              | Primary metric                         | Bar         |
| ------------------ | -------------------------------------- | ----------- |
| Memorize (scratch) | Teacher `ordered_onset_match` @ 20 ms  | **634/634** |
| Decode             | Free-run `ordered_onset_match` @ 20 ms | **634/634** |

**Scratch-only:** `init_model_path` stripped every run. Warm-start 614/634 is **not** a valid pass bar.

**Fixed eval (do not change):** teacher gate before `--ar_decode`, checkpoint `val_overfit_gate`, offline eval only.

## Preflight

```text
# Kill stray drivers (Windows)
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
  Where-Object { $_.CommandLine -match 'run_overnight|run_exp|train_onset_ar' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

wsl bash -lc "pkill -f 'train_onset_ar\.py' || true"

venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'scripts/ar_tide_iter'); import training_lock; training_lock.clear_stale_lock()"
```

- **Never** `run_overnight.py --hours` (planner / knob lattice).
- **One** `run_exp` / training job at a time (`training_lock.py`).

## Orient

```text
venv\Scripts\python.exe scripts/ar_tide_iter/session_brief.py
```

Also skim [AR_TIDE_OVERFIT_ITER_LOG.md](../../../../docs/research/AR_TIDE_OVERFIT_ITER_LOG.md) for qualitative learnings.

Key brief fields: `session_best`, scratch teacher best (iter43+ / notes “Scratch”), `last_run_vs_session_best`, `tried_recipes`, `config_changes_vs_previous`.

## Plan artifact

`logs/ar_tide_iter/next_experiment.json`:

```json
{
  "id": "iterNNN",
  "notes": "One-line hypothesis",
  "reasoning": "Evidence; tier; expected metric movement",
  "run": {}
}
```

Partial overrides only (`run` / `model` / `dataset`). Example: [next_experiment.example.json](../../../../scripts/ar_tide_iter/next_experiment.example.json).

**Pinned keys:** `epochs=400`, `checkpoint_metric=val_overfit_gate`, `tolerance_sec=0.02` unless champion template changes.

## Run

### Single iteration

```text
# Agent writes next_experiment.json first:
venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --once

# Machine-readable footer (also printed with --autoresearch):
venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --once --json
```

### Long budget sessions (required when user gives hours)

**Use the full wall-clock budget** until free-run **634/634** or `now >= deadline`. ~8–10 min per run → **7 h ≈ 40–50 iterations**, not a short pre-written list.

**Agent-in-loop (default):** repeat until deadline or goal:

```text
deadline = now + budget_hours
while now < deadline and not goal_passed:
    venv\Scripts\python.exe scripts/ar_tide_iter/session_brief.py
    # write logs/ar_tide_iter/next_experiment.json from evidence
    venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --once --brief none
    # log AR_TIDE_OVERFIT_ITER_LOG.md; replan — never stop because a seed queue emptied
```

**Harness-assisted (optional):** start the waiter, but **keep feeding plans**:

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --hours 7 --plan-wait 900
```

Write the next `next_experiment.json` after each run **before** `plan-wait` expires. If you background this process, monitor it and supply plans until deadline.

**Forbidden for budgeted autoresearch:**

- Custom scripts that iterate a fixed `experiment_queue.json` and exit when empty
- Treating “queue finished” as session complete while time remains
- Bare `--hours` without `--autoresearch` (enables `overnight_planner` lattice)

`--autoresearch` remaps exit code **0** when train+eval finished but teacher/free-run gate not met; **1** only for infra/plan failures. Look for `=== AUTORESEARCH_SUMMARY ===` in output.

Watch: `venv\Scripts\python.exe scripts/ar_tide_iter/show_status.py --id iterNNN --watch`

## Tier C (AR code)

```text
venv\Scripts\python.exe -m pytest tests/onset_ar/ -q --tb=short
venv\Scripts\python.exe pre_submit.py --fast
```

## Log

| What                 | Where                                                                                |
| -------------------- | ------------------------------------------------------------------------------------ |
| Every iter run       | [AR_TIDE_OVERFIT_ITER_LOG.md](../../../../docs/research/AR_TIDE_OVERFIT_ITER_LOG.md) |
| Graduation / bug fix | [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md)                     |

Do **not** commit `logs/`.

## Profile anti-spam

1. Scratch teacher plateau **≥5 runs** (±10 matches) → no single-key `learning_rate` / `lambda_*` / `eos_token_weight_scale` tweaks; upgrade tier or hypothesis class.
2. Teacher **&lt; 0.95** → no scheduled sampling, in-loop AR decode, or decode checkpoint metrics.
3. `val_ordered_onset_match` **&lt; 0.5** @ ~30 ep → pivot recipe; do not burn 200 ep.

## Phase detection

| Signal                             | Focus                                        |
| ---------------------------------- | -------------------------------------------- |
| Scratch teacher &lt; 634/634       | Memorization recipe                          |
| Teacher 634/634, free-run &lt; 634 | Decode exposure / consistency                |
| Flat val gate early                | Capacity (Tier B) or implementation (Tier C) |

## Hypothesis classes (when plateaued)

| Class                 | Direction                              |
| --------------------- | -------------------------------------- |
| Champion memorization | `lr=5e-5`, no SS, `λ_inc=0.1`, 200 ep  |
| High-lr memorization  | `lr=1e-4`, no SS                       |
| Capacity              | `d_model`, layers, `patch_frames`      |
| Decode polish         | mild SS, `λ_inc` after teacher perfect |
| Implementation        | val/decode/loss bugs + tests           |

Low-yield repeats: heavy SS `p≥0.4`, `use_soft_pointer_time`, warm-start polish.

## Success actions

Free-run **634/634** → `graduate_ar_tide_overfit.py` → stop.

## Gotchas (ar-tide-overfit)

| Gotcha                              | What goes wrong                                                                                        | What to do                                                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **`init_model_path` / warm-start**  | Fast teacher or free-run that does not count for scratch gate                                          | Harness strips it; never plan warm-start; 614/634 warm-start is not a pass bar                                    |
| **`overnight_planner` / `--hours`** | Copies `epochs=150`, `checkpoint_metric=val_ar_decode_*`, single-key mutations from old warm-start era | Agent autoresearch only: `--once` + `next_experiment.json`                                                        |
| **Stale `gpu_training.lock`**       | Dead Windows PID blocks start, or lock cleared while WSL still training → duplicate iter               | `training_lock.clear_stale_lock()`; `pgrep train_onset_ar` in WSL; never delete lock manually while a job is live |
| **Zombie `run_overnight` shells**   | Second driver plans duplicate `iterNN` while first train runs                                          | Kill all `run_overnight`/`run_exp` in preflight; one driver                                                       |
| **Decode knobs before memorize**    | SS / `lambda_inc` / in-loop AR decode while teacher &lt; 0.95                                          | Memorization recipe first (`lr`, no SS, full 200 ep)                                                              |
| **Teacher perfect but wrong f1**    | `ordered=634/634` but `event_f1&lt;1` → free-run skipped                                               | Fix training objective or accept gate fail; do not claim pass                                                     |
| **Pinned keys drift**               | `epochs≠400`, wrong `checkpoint_metric`, `tolerance_sec` changed by planner                            | Keep champion pins unless template/design doc changes                                                             |
| **Registry vs results**             | `experiments.json` has id N while `results.jsonl` last is N−1; in-flight train                         | `next_iter_id` uses registry; wait for train before re-planning same id                                           |
| **Infra vs recipe failure**         | `train_exit≠0` or missing checkpoint                                                                   | Retry same id `--reuse-last-config`; new recipe → new id                                                          |
| **Val gate flat @ ~30 ep**          | Recipe not learning tide                                                                               | Pivot; do not burn 200 ep (see overnight handoff heuristics)                                                      |
| **Session best ≠ scratch best**     | Brief shows 614/634 free-run from iter17 warm-start                                                    | Rank scratch-era teacher (iter43+, notes “Scratch”) for memorize phase                                            |
| **Windows process pairs**           | Two `python.exe` PIDs per job (venv + launcher)                                                        | Normal if one WSL `train_onset_ar`; verify with `pgrep`, not PID count alone                                      |
| **Finite experiment queue**         | 8 pre-written plans finish in ~1 h; 7 h budget unused                                                  | [Budget discipline](../../SKILL.md#budget-discipline): replan after each run until deadline or 634/634 free-run |

## Harness

[scripts/ar_tide_iter/README.md](../../../../scripts/ar_tide_iter/README.md)
