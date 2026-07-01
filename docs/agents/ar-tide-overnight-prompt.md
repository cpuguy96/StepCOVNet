# AR tide overfit — overnight research handoff

**Paste to a new agent:** everything in [§ Agent prompt](#agent-prompt) below.

| Item | Value |
| ---- | ----- |
| **Goal** | Free-run **634/634** ordered onset match @ 20 ms on tide (scratch train, no cheating) |
| **Time budget** | ~7 hours unattended |
| **Scratch policy** | **Random init every run** — `init_model_path` is stripped; loading prior checkpoints is **cheating** |
| **Last run** | `iter43` — scratch, teacher **56/634** @ 200 ep (SS=0.2, lr=2e-5) |
| **Next experiment id** | `iter44` (agent picks via `session_brief.py`) |
| **Harness** | `scripts/ar_tide_iter/` · log to `docs/research/AR_TIDE_OVERFIT_ITER_LOG.md` |
| **Do not commit** | Anything under `logs/` |

**Read first:** `AGENTS.md` → [docs/research/README.md](../research/README.md), then [AR_ONSET_DESIGN.md](../research/AR_ONSET_DESIGN.md), [AR_TIDE_OVERFIT_ITER_LOG.md](../research/AR_TIDE_OVERFIT_ITER_LOG.md).

---

## Agent prompt

You are an autonomous research agent on **StepCOVNet** (`master`). Your job for the next **~7 hours** is to close the gap on **free-run AR decode** for the single-chart tide overfit gate — **634/634 ordered onset matches @ 20 ms** — with **no cheating**.

### North star

| Metric | Bar |
|--------|-----|
| **Primary** | `ar_decode.ordered_onset_match` = **634/634 @ 20 ms** vs **`target_times`** (free-run, two-pass decode) |
| **Aux** | `chart_ordered_onset_match` vs raw chart `gt_times`; Hungarian `event_f1` |
| Teacher | **634/634** vs `target_times` — prerequisite before free-run eval |

**Failure mode (known):** not early EOS (decode length ~636). Free-run misses are **residual timing drift** during autoregressive decode — patches often right, cumulative `residual_err_ms` blows past 20 ms on ~20 steps. See diagnostics in [AR_TIDE_OVERFIT_ITER_LOG.md](../research/AR_TIDE_OVERFIT_ITER_LOG.md).

### No cheating (hard rules)

Do **not** count success from:

- **`init_model_path` / warm-start** — every tide iter run must memorize tide from **random init** on that single chart. Prior checkpoint weights are not allowed (harness strips `init_model_path`). Historical 614/634 runs that warm-started are **not** valid pass bars.
- Teacher-forced / gold-prefix decode at eval time
- `min length = n_gt` or other train-time length hacks (debug only per design doc)
- `use_soft_pointer_time: true` or anything that leaks GT timing into decode
- Claiming pass on `val_event_onset_f1` or Hungarian F1 alone — **ordered onset match** is primary ([ONSET_METRICS.md](../research/ONSET_METRICS.md))
- Checkpointing on teacher (`val_overfit_gate`) while reporting free-run wins from a different checkpoint

Scheduled sampling during **training** is allowed. Inference must be **pure free-run** `ar_decode`.

### Code changes (hard rule — before any GPU training)

Any change to **`src/`**, **`scripts/`** (other than iteration JSON configs under `logs/`), or training/eval behavior:

1. **Add or update tests** that cover the changed behavior (prefer `tests/onset_ar/` for AR stack).
2. **Run and pass** targeted pytest **before** starting a WSL training job:

```text
venv\Scripts\python.exe -m pytest tests/onset_ar/ -q --tb=short
venv\Scripts\python.exe pre_submit.py --fast
```

3. **Do not train** on unverified code — a 200-epoch run on a broken decode path wastes the whole GPU window.

Config-only experiments (hyperparameters, loss weights in JSON) need no new tests. **Logic fixes** (decode, losses, callbacks, metrics) always do. Tide iteration never warm-starts (`init_model_path` is stripped).

### Research eval policy (overnight iteration)

Same as champion — **offline AR decode only** (~0.5s val/epoch during training):

- **No in-loop free-run decode** during training (removed `ArDecodeValidationCallback`)
- **Checkpoint on teacher:** `checkpoint_metric: val_overfit_gate`
- **Early stop on teacher:** `perfect_overfit_early_stop: true`, `perfect_overfit_min_score: 0.999`, `patience: 5`
- **Offline eval** (`debug_ar_onset_overfit.py --ar_decode`) after every run; on session bests before graduate/declare pass

### Experiment budget

| Constraint | Value |
|------------|-------|
| Max epochs per run | **400** |
| Early stop | Use when teacher is **climbing** and plateaus; **do not** early-stop a run still below teacher **0.99** on a single-song overfit |
| GPU | **One job at a time** — file lock + `pgrep train_onset_ar` in WSL (see `training_lock.py`) |
| Runs | Prefer **many experiments** over one long run; **never** leave old `run_overnight` shells running |

**Kill / pivot heuristics (scratch):**

- If `val_ordered_onset_match` **&lt; 0.5** after **~30 epochs**, stop and change recipe (iter43-style SS+low lr failed to memorize)
- If teacher ≥ **0.999** but flat **15–20 epochs**, pivot decode/exposure recipe (e.g. add mild SS), not more epochs alone
- If teacher ≥ **0.999** but offline free-run stuck **&lt; 0.97**, decode-path / exposure-bias issue — fix or tune `lambda_inc`, not warm-start

### Scratch-only training (mandatory)

Single-chart overfit **must** work from random initialization — that is the point of the tide gate. The harness merges overrides onto `configs/ar/tide_overfit.json` and **always removes** `init_model_path`.

**First priority on scratch:** reach teacher **634/634** (memorize the chart) before tuning free-run decode. Typical order:

1. **Memorize** — champion-like recipe: `learning_rate` **5e-5** (or **1e-4** if slow), **`scheduled_sampling_max_p: 0`**, `perfect_overfit_early_stop: false` until teacher perfect
2. **Decode** — add mild scheduled sampling / `lambda_incremental_consistency` only after teacher is perfect

Do **not** copy warm-start checkpoint paths from old log entries.

### Hypotheses

Use **session_brief** config diffs (`config_changes_vs_previous`, `last_run_vs_session_best`) and free-run scores — not a fixed knob checklist. Any key in `configs/ar/tide_overfit.json` or a prior snapshot is fair game (`run`, `model`, `dataset`). When code adds a new training feature, the new config key appears in the template and brief automatically.

**Eval metrics stay fixed** across runs: free-run `ordered_onset_match` @ 20 ms, teacher gate before `--ar_decode`, `val_overfit_gate` for checkpointing. Compare runs only on those metrics.

**Already tried (low yield):** heavy SS (`p=0.4`), `use_soft_pointer_time`, continuing iter17 without new recipe (iter22/23), iter17+offline-only ckpt alone (iter30 → 612/634).

### How to run

**One prompt (keep Cursor open — recommended):**

```text
Run autoresearch.
Profile: ar-tide-overfit
Goal: scratch teacher 634/634 then free-run 634/634 on tide @ 20 ms.
Budget: 7 hours.
Go — do not ask me between runs.
```

Agent follows [.cursor/skills/autoresearch/SKILL.md](../../.cursor/skills/autoresearch/SKILL.md) + [ar-tide-overfit profile](../../.cursor/skills/autoresearch/profiles/ar-tide-overfit.md). **Not** bare `run_overnight --hours` (use `--autoresearch --hours` or `--allow-planner`).

**Budget:** use the **full** wall-clock budget until goal or deadline — see skill § Budget discipline (~40–50 tide runs per 7 h). Do not stop when a pre-written queue empties.

**Agent autoresearch (manual steps):**

Load [.cursor/skills/autoresearch/SKILL.md](../../.cursor/skills/autoresearch/SKILL.md), then each cycle:

```text
venv\Scripts\python.exe scripts/ar_tide_iter/session_brief.py
```

Write `logs/ar_tide_iter/next_experiment.json` from evidence (not knob spam), then:

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --once --brief none
```

Repeat until time budget or **634/634** free-run. **Do not** use bare `--hours` — that is blocked unless you pass `--allow-planner` (history lattice / single-key tweaks only).

**Unattended JSON search only** (low exploration — use when no agent is available):

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --hours 7 --allow-planner
```

`overnight_planner.py` ranks session runs and mutates config neighbors. One GPU job at a time.

**Single experiment** (when you already registered the recipe in `experiments.json`):

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --id iter31 ^
    --notes "your hypothesis"
```

Add `iter31+` entries in `scripts/ar_tide_iter/experiments.json` (new id per hypothesis). Retries reuse the same id — see `experiments.README.md`. Use `--reuse-last-config` for infra-only retries; edit the registry only when the recipe changes.

**Watch progress (logs/ is gitignored):**

```text
venv\Scripts\python.exe scripts/ar_tide_iter/show_status.py --id iter31 --watch
```

Run `run_exp.py` in the **foreground** terminal if you want live epoch lines; background runs are silent except via `show_status`.

**Offline confirmation:**

```text
venv\Scripts\python.exe scripts/debug_ar_onset_overfit.py ^
    --config logs/ar_tide_iter/configs/iter31.json ^
    --model_path models_wsl/ar/tide_overfit_iter/iter31/ar_onset_model.keras ^
    --ar_decode --json-only
```

**Graduate only on offline win beating champion:**

```text
venv\Scripts\python.exe scripts/graduate_ar_tide_overfit.py ^
    --config configs/ar/versions/tide_overfit/vN.json ^
    --model-path models_wsl/.../ar_onset_model.keras ^
    --version-ref configs/ar/versions/tide_overfit/vN.json
```

### Logging and commits

| Write to | What |
|----------|------|
| `docs/research/AR_TIDE_OVERFIT_ITER_LOG.md` | Every experiment (hypothesis, metrics, learnings) |
| `docs/research/EXPERIMENT_LOG.md` | Bug fixes or graduation only |
| `logs/` | Machine output — **never commit** |

**Commits:** code fixes **only with passing tests** (`pytest` + `pre_submit.py --fast`); batch research-log updates every few hours or on session best. No `logs/`, no checkpoints.

### Loop (repeat until time or 634/634)

**You** choose each experiment. Do not march a fixed `iter34→iter38` queue and do not rely on Python to auto-mutate hyperparameters.

1. `session_brief.py` — read session best, config diffs vs prior runs, val metrics, tried recipes
2. Read `AR_TIDE_OVERFIT_ITER_LOG.md` for qualitative learnings
3. **Decide** from diffs what to change next; write `next_experiment.json` with only those overrides plus `reasoning`
4. **If code changed:** tests → `pre_submit.py --fast` → then train
5. `run_overnight.py --autoresearch --once` — trains/evals the plan you wrote
6. Offline free-run only when teacher is perfect (automatic in `run_exp.py`)
7. **634/634** free-run → `graduate_ar_tide_overfit.py` → stop

Example reasoning: iter43 scratch with SS=0.2 and lr=2e-5 only reached 56/634 teacher — iter44 returns to champion memorization defaults (lr=5e-5, no SS, full 200 ep) before any decode knobs.

### Stop conditions

- **Success:** offline **634/634** → graduate, update EXPERIMENT_LOG, stop
- **Time:** write session summary at top of `AR_TIDE_OVERFIT_ITER_LOG.md`
- **Blocked:** document blocker; do not spin silently

### Environment

- Repo root: `venv\Scripts\python.exe` (auto WSL GPU dispatch)
- GPU override: `STEPCOVNET_FORCE_GPU=1` or `run_exp.py --force`
- Skills: [.cursor/skills/wsl-gpu-stepcovnet/SKILL.md](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md)

**Mandate:** single-song overfit from **scratch** must reach teacher 634/634, then close free-run decode gap — no checkpoint chaining.
