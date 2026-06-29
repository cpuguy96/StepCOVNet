# AR tide overfit — overnight research handoff

**Paste to a new agent:** everything in [§ Agent prompt](#agent-prompt) below.

| Item | Value |
| ---- | ----- |
| **Goal** | Free-run **634/634** ordered onset match @ 20 ms on tide (no cheating) |
| **Time budget** | ~7 hours unattended |
| **Session best (offline)** | **614/634** — `iter17` / `iter18` / `iter21` (tied) |
| **Last run** | `iter30` — offline **612/634** (iter17 recipe, teacher ckpt only) |
| **Warm-start** | `models_wsl/ar/tide_overfit_iter/iter17/ar_onset_model.keras` |
| **Next experiment id** | `iter31` |
| **Harness** | `scripts/ar_tide_iter/` · log to `docs/research/AR_TIDE_OVERFIT_ITER_LOG.md` |
| **Do not commit** | Anything under `logs/` |

**Read first:** `AGENTS.md` → [docs/research/README.md](../research/README.md), then [AR_ONSET_DESIGN.md](../research/AR_ONSET_DESIGN.md), [AR_TIDE_OVERFIT_ITER_LOG.md](../research/AR_TIDE_OVERFIT_ITER_LOG.md).

---

## Agent prompt

You are an autonomous research agent on **StepCOVNet** (`master`). Your job for the next **~7 hours** is to close the gap on **free-run AR decode** for the single-chart tide overfit gate — **634/634 ordered onset matches @ 20 ms** — with **no cheating**.

### North star

| Metric | Bar |
|--------|-----|
| **Primary** | `ar_decode.ordered_onset_match` = **634/634 @ 20 ms** on tide (free-run, two-pass decode) |
| Teacher | Already **634/634** — not the bottleneck |

**Failure mode (known):** not early EOS (decode length ~636). Free-run misses are **residual timing drift** during autoregressive decode — patches often right, cumulative `residual_err_ms` blows past 20 ms on ~20 steps. See diagnostics in [AR_TIDE_OVERFIT_ITER_LOG.md](../research/AR_TIDE_OVERFIT_ITER_LOG.md).

### No cheating (hard rules)

Do **not** count success from:

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

3. **Do not train** on unverified code — a 150-epoch run on a broken decode path wastes the whole GPU window.

Config-only experiments (hyperparameters, init checkpoint, loss weights in JSON) need no new tests. **Logic fixes** (decode, losses, callbacks, metrics) always do.

### Research eval policy (overnight iteration)

For **this research loop** (differs from champion `configs/ar/tide_overfit.json`):

- **Use in-loop AR decode** for fast feedback: `ar_decode_val_every_n_epochs: 5` or `10`
- **Checkpoint on free-run:** `checkpoint_metric: val_ar_decode_ordered_onset_match`
- **Early stop on free-run:** `perfect_overfit_early_stop: true`, `perfect_overfit_min_score: 0.999`, `patience: 5`
- **Offline eval** (`debug_ar_onset_overfit.py --ar_decode`) when in-loop hits a new session best, or before graduate/declare pass

Champion config stays offline-only; iteration configs may use in-loop decode.

### Experiment budget

| Constraint | Value |
|------------|-------|
| Max epochs per run | **150** |
| Early stop | **Required** on flat learning — do not burn 150 ep on dead configs |
| GPU | **One WSL job at a time** — `wsl_gpu.assert_wsl_gpu_free_for_training()` is automatic |
| Runs | Prefer **many small experiments** over one long run |

**Kill / pivot heuristics:**

- If `val_ar_decode_ordered_onset_match` flat **15–20 epochs** and below **0.97**, stop and change hypothesis
- If teacher ≥ 0.998 but free-run stuck &lt; 0.97 after ~40 epochs → exposure-bias / decode-path issue, not more epochs

### Warm starts

| Checkpoint | Free-run (offline) | Notes |
|------------|-------------------|-------|
| `models_wsl/ar/tide_overfit_iter/iter17/ar_onset_model.keras` | **614/634** | `λ_inc=0.2`, `max_steps=32`, `lr=2e-5` |
| `models_wsl/ar/tide_overfit_iter/iter18/ar_onset_model.keras` | **614/634** | `λ_residual=20` |
| `models_wsl/ar/tide_overfit_iter/iter21/ar_onset_model.keras` | **614/634** | mild scheduled sampling |
| `models_wsl/ar/tide_overfit/ar_onset_model.keras` | ~611–619 | champion baseline |

Default init for new runs: **iter17** unless hypothesis needs otherwise.

### Hypotheses (priority order)

1. **Decode-path bugs** — compare teacher vs `ar_decode` in `src/stepcovnet/onset_ar/`; fix + test if real bug found
2. **Incremental consistency** — `λ_incremental_consistency` 0.2–0.35, `incremental_consistency_max_steps` 32–64
3. **Residual head** — `lambda_residual` 15–30 with tuned `lr`
4. **Scheduled sampling** — `scheduled_sampling_max_p` 0.1–0.3, warmup 50, ramp 80 (from iter17)
5. **EOS / length** — `eos_token_weight_scale` (secondary; EOS not current failure mode)

**Already tried (low yield):** heavy SS (`p=0.4`), `use_soft_pointer_time`, continuing iter17 without new recipe (iter22/23), iter17+offline-only ckpt alone (iter30 → 612/634).

### How to run

```text
venv\Scripts\python.exe scripts/ar_tide_iter/build_configs.py
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --id iter31 ^
    --config logs/ar_tide_iter/configs/iter31.json ^
    --notes "your hypothesis"
```

Add `iter31+` entries in `scripts/ar_tide_iter/build_configs.py` before `build_configs.py`. For overnight iteration, set in-loop decode + free-run checkpoint in each new config.

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

1. GPU free → one hypothesis → `iterXX.json` (in-loop decode, free-run ckpt)
2. **If code changed:** write tests → `pytest` → `pre_submit.py --fast` → then train
3. Train ≤150 ep, early stop if flat
4. Record in-loop peak `val_ar_decode_ordered_onset_match`
5. New session best → offline `--ar_decode`
6. Offline 634/634 → `graduate_ar_tide_overfit.py` → stop

### Stop conditions

- **Success:** offline **634/634** → graduate, update EXPERIMENT_LOG, stop
- **Time:** write session summary at top of `AR_TIDE_OVERFIT_ITER_LOG.md`
- **Blocked:** document blocker; do not spin silently

### Environment

- Repo root: `venv\Scripts\python.exe` (auto WSL GPU dispatch)
- GPU override: `STEPCOVNET_FORCE_GPU=1` or `run_exp.py --force`
- Skills: [.cursor/skills/wsl-gpu-stepcovnet/SKILL.md](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md)

**Mandate:** debug + research, not blind hyperparameter spam. Teacher path proves representational capacity — the gap is **autoregressive decode fidelity**.
