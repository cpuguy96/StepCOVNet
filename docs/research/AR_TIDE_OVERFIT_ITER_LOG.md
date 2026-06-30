# AR tide overfit iteration log (agent session)

Goal: free-run ordered **634/634 @ 20 ms**. Machine logs: `logs/ar_tide_iter/` (gitignored).

Started: 2026-06-28T19:37:03 · **Last updated:** 2026-06-30 (autoresearch 7h — iter174+)

## Autoresearch session (2026-06-30, 7h)

| | |
|--|--|
| **Start** | iter174 — scratch replay iter169 (d_model=384, 633/634 teacher best) |
| **Completed** | iter174–182 (queue resumed after BOM fix) |
| **Scratch best (ordered)** | **iter175, iter178 — teacher 634/634** (first scratch perfect ordered) |
| **Scratch best (event_f1)** | **0.9984** — gate still blocks free-run (needs event_f1=1.0) |
| **Winning tweaks** | `lambda_residual=30` (iter175); `lambda_time=1.5` + `lambda_res=28` (iter178) |
| **Rejected** | d_model=320 (632), d_model=448 (630), 5-layer depth (OOM), mild SS (624), early stop (633) |

## Overnight session (2026-06-29, second pass)

| | |
|--|--|
| **Start** | iter34 — warm-start `iter17` (614/634 session best) |
| **Last completed** | iter33 — teacher **634/634**, free-run **614/634** (tied best) |
| **Goal** | Free-run **634/634 @ 20 ms** |
| **Budget** | ~7 h unattended |
| **Queue** | iter34 → iter38 (`experiments.json`) |

## Overnight session (2026-06-29, resumed)

| | |
|--|--|
| **Start** | iter32 — warm-start `iter17` (614/634 session best) |
| **Last completed** | iter31 attempt 2 — **610/634** offline |
| **Goal** | Free-run **634/634 @ 20 ms** |
| **Budget** | ~7 h unattended |

## Overnight session (2026-06-29, first pass)

| | |
|--|--|
| **Start** | iter31 — warm-start `iter17` (614/634); offline-only training eval |
| **Goal** | Free-run **634/634 @ 20 ms** |
| **Budget** | ~7 h unattended |

## Session summary (closed 2026-06-29 — prior handoff)

| | |
|--|--|
| **Best free-run (offline)** | **614/634 (96.85%)** — `iter17`, `iter18`, `iter21` (tied) |
| **Best checkpoint** | `models_wsl/ar/tide_overfit_iter/iter17/ar_onset_model.keras` |
| **Champion baseline** | 611/634 offline re-eval (manifest lists 619/634) |
| **Experiments run** | iter01–iter30 (iter29 aborted; iter30 offline-only → 612/634) |
| **Goal** | **Not reached** — ~20 onsets miss @ 20 ms in free-run |
| **Overnight handoff** | [docs/agents/ar-tide-overnight-prompt.md](../agents/ar-tide-overnight-prompt.md) — start at **iter31** |

## Prior session notes (iter01–iter27)

### Failure mode (diagnostics on iter01)

- Teacher path: **634/634** perfect ordered match; patches and residuals correct.
- Free-run: **634 predictions**, **613 ordered matches** — not early EOS (decode length 636, EOS yes).
- Gap is **accumulated residual timing drift** during autoregressive decode (patches often correct, `residual_err_ms` > 20 ms on worst steps).
- Teacher early-stop on `val_overfit_gate` fires ~epoch 51 while free-run still ~608; disabling early stop helps reach 613 but not 634.

### What moved the needle (+3 vs champion)

1. **iter01** — resume champion + short polish → 613/634 (early stop @ teacher gate).
2. **iter17** — `λ_inc=0.2`, `max_steps=32`, in-loop AR decode val every 10ep, `lr=2e-5` → **614/634**.
3. **iter18** — `λ_residual=20`, `lr=2e-5` from iter01 → **614/634** (tie).
4. **iter21** — mild scheduled sampling (`p→0.2`, warmup 50) + `λ_inc=0.15` → **614/634** (tie).

### What did not help (plateau ~611–613)

- Higher `λ_inc` (0.3–0.4), dropping `λ_inc`, `eos_weight_scale=0.2`, warm-start run2 alone.
- Full 150ep without early stop (iter08/11), heavy SS (`p=0.4`), `use_soft_pointer_time`.
- Combining best recipes (iter25/26), continuing from iter17 (iter22/23).

### Infra notes

- WSL `pkg_resources` missing mid-session — fixed with `pip install setuptools` in `~/stepcovnet-venv-wsl`.
- **Fixed (2026-06-29):** in-loop `val_ar_decode_ordered_onset_match` logged **0.0** because `test_step` reset/published AR-decode metrics before `ArDecodeValidationCallback` ran — broke checkpoint-on-free-run. See `src/stepcovnet/onset_ar/trainers.py` (`_batch_metrics` excludes callback-only metrics).
- Harness: `scripts/ar_tide_iter/run_exp.py`, `scripts/ar_tide_iter/experiments.json` (outputs under `logs/ar_tide_iter/`).

---

### baseline (2026-06-28T19:37:03)

**Hypothesis:** Champion checkpoint baseline before iteration session

|                  |                                                   |
| ---------------- | ------------------------------------------------- |
| Config           | `configs\ar\tide_overfit.json`                    |
| Model            | `models_wsl\ar\tide_overfit\ar_onset_model.keras` |
| Train exit       | skipped                                           |
| Train log        | `logs\ar_tide_iter\train_logs\baseline.log`       |
| Teacher ordered  | 633/634 (0.9984)                                  |
| Free-run ordered | **611/634 (0.9637)**                              |
| Decode steps     | 636                                               |
| Eval wall (s)    | 101.83                                            |

### iter01 (2026-06-28T19:40:47)

**Hypothesis:** Resume champion weights; 150ep; early stop @ val_overfit_gate 0.9999

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter01.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter01\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter01.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 91.2                                                          |

### iter02 (2026-06-28T19:43:35)

**Hypothesis:** Warm-start run2; v3 polish lambda_res=10, no lambda_inc, lr=1e-4

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter02.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter02\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter02.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **581/634 (0.9164)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 96.47                                                         |

### iter03 (2026-06-28T19:46:23)

**Hypothesis:** Resume champion; lower lambda_inc=0.01; lambda_res=12

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter03.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter03\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter03.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **610/634 (0.9621)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 96.57                                                         |

### iter05 (2026-06-28T19:48:57)

**Hypothesis:** Resume champion; drop lambda_inc; lambda_res=15; lr=2e-5

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter05.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter05\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter05.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **611/634 (0.9637)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 91.69                                                         |

### iter06 (2026-06-28T19:52:01)

**Hypothesis:** Warm-start run2; eos_token_weight_scale=0.2

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter06.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter06\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter06.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **608/634 (0.9590)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 93.29                                                         |

### iter04 (2026-06-28T19:56:08)

**Hypothesis:** Warm-start run2; higher lambda_inc=0.25

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter04.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter04\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter04.log`                     |
| Teacher ordered  | 632/634 (0.9968)                                              |
| Free-run ordered | **611/634 (0.9637)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 97.21                                                         |

### iter08 (2026-06-28T20:00:48)

**Hypothesis:** Resume iter01; NO early stop; full 150ep

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter08.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter08\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter08.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 94.09                                                         |

### iter09 (2026-06-28T20:05:07)

**Hypothesis:** Resume iter01; no early stop; lambda_inc=0.2; lr=3e-5

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter09.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter09\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter09.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **608/634 (0.9590)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 94.58                                                         |

### iter10 (2026-06-28T20:09:31)

**Hypothesis:** Resume iter01; no early stop; lambda_inc=0.3; lambda_res=12

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter10.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter10\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter10.log`                     |
| Teacher ordered  | 633/634 (0.9984)                                              |
| Free-run ordered | **612/634 (0.9653)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 95.95                                                         |

### iter11 (2026-06-28T20:09:41)

**Hypothesis:** Resume champion; no early stop; full 150ep

|            |                                                               |
| ---------- | ------------------------------------------------------------- |
| Config     | `logs\ar_tide_iter\configs\iter11.json`                       |
| Model      | `models_wsl\ar\tide_overfit_iter\iter11\ar_onset_model.keras` |
| Train exit | 1                                                             |
| Train log  | `logs\ar_tide_iter\train_logs\iter11.log`                     |
| Error      | checkpoint missing after train                                |

### iter11 (2026-06-28T20:21:01)

**Hypothesis:** Resume champion; no early stop; full 150ep

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter11.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter11\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter11.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 93.56                                                         |

### iter12 (2026-06-28T20:25:23)

**Hypothesis:** Resume iter01; no early stop; incremental_consistency_max_steps=32

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter12.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter12\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter12.log`                     |
| Teacher ordered  | 633/634 (0.9984)                                              |
| Free-run ordered | **607/634 (0.9574)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 96.66                                                         |

### iter13 (2026-06-28T21:04:55)

**Hypothesis:** Resume iter01; in-loop AR decode val every 5ep; ckpt free-run metric

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter13.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter13\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter13.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 83.02                                                         |

### iter14 (2026-06-28T21:08:41)

**Hypothesis:** Resume iter01; scheduled sampling p to 0.4 over 120ep

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter14.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter14\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter14.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **611/634 (0.9637)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 85.45                                                         |

### iter15 (2026-06-28T21:12:29)

**Hypothesis:** Resume iter01; heavy lambda_inc=0.4; max_steps=64

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter15.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter15\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter15.log`                     |
| Teacher ordered  | 633/634 (0.9984)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 85.21                                                         |

### iter17 (2026-06-28T21:33:15)

**Hypothesis:** Resume iter01; lambda_inc=0.2 + AR decode every 10ep

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter17.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter17\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter17.log`                     |
| Teacher ordered  | 633/634 (0.9984)                                              |
| Free-run ordered | **614/634 (0.9685)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 86.88                                                         |

### iter18 (2026-06-28T21:37:14)

**Hypothesis:** Resume iter01; lambda_residual=20

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter18.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter18\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter18.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **614/634 (0.9685)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 84.97                                                         |

### iter22 (2026-06-28T21:40:59)

**Hypothesis:** Resume iter17 best; full 150ep polish

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter22.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter22\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter22.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 90.44                                                         |

### iter19 (2026-06-28T21:45:27)

**Hypothesis:** Resume iter01; soft pointer time + lambda_inc=0.2

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter19.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter19\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter19.log`                     |
| Teacher ordered  | 630/634 (0.9937)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 111.54                                                        |

### iter23 (2026-06-28T21:50:06)

**Hypothesis:** Resume iter17; higher lambda_inc=0.25 max_steps=48

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter23.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter23\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter23.log`                     |
| Teacher ordered  | 633/634 (0.9984)                                              |
| Free-run ordered | **611/634 (0.9637)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 111.96                                                        |

### iter20 (2026-06-28T21:54:42)

**Hypothesis:** Resume iter01; ultra-low lr=5e-6 micro-polish

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter20.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter20\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter20.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **612/634 (0.9653)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 111.23                                                        |

### iter25 (2026-06-28T21:59:32)

**Hypothesis:** Resume iter17; combine lambda_inc=0.2 + lambda_res=20

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter25.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter25\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter25.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **611/634 (0.9637)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 108.58                                                        |

### iter26 (2026-06-28T22:03:58)

**Hypothesis:** Resume iter18; add lambda_inc=0.2

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter26.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter26\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter26.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **613/634 (0.9669)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 108.79                                                        |

### iter27 (2026-06-28T22:09:14)

**Hypothesis:** Resume iter17; lambda_res=15

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter27.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter27\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter27.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **612/634 (0.9653)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 107.54                                                        |

### iter21 (2026-06-28T22:13:51)

**Hypothesis:** Resume iter01; mild SS p=0.2 + lambda_inc=0.15

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter21.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter21\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter21.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **614/634 (0.9685)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 109.74                                                        |

### iter28 (2026-06-28T22:19:11)

**Hypothesis:** Resume iter21; add lambda_res=20 combo polish

|                  |                                                               |
| ---------------- | ------------------------------------------------------------- |
| Config           | `logs\ar_tide_iter\configs\iter28.json`                       |
| Model            | `models_wsl\ar\tide_overfit_iter\iter28\ar_onset_model.keras` |
| Train exit       | 0                                                             |
| Train log        | `logs\ar_tide_iter\train_logs\iter28.log`                     |
| Teacher ordered  | 634/634 (1.0000)                                              |
| Free-run ordered | **611/634 (0.9637)**                                          |
| Decode steps     | 636                                                           |
| Eval wall (s)    | 103.72                                                        |

### iter29 · attempt 1 (2026-06-29T02:19:16)

**Hypothesis:** iter17 recipe retry post trainer fix (AR decode ckpt)

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Config | `logs\ar_tide_iter\configs\iter29.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter29\ar_onset_model.keras` |
| Train exit | 15 |
| Train log | `logs\ar_tide_iter\train_logs\iter29.log` |
| Error | checkpoint missing after train |

### iter29 · attempt 2 (2026-06-29T02:28:26)

**Hypothesis:** iter17 recipe retry post trainer fix (AR decode ckpt)

| | |
|--|--|
| Kind | retry |
| Attempt | 2 |
| Config | `logs\ar_tide_iter\configs\iter29.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter29\ar_onset_model.keras` |
| Train exit | 15 |
| Train log | `logs\ar_tide_iter\train_logs\iter29.log` |
| Error | checkpoint missing after train |

### iter30 (2026-06-29T02:32:39)

**Hypothesis:** iter17 recipe, offline decode only (replaces iter29)

| | |
|--|--|
| Config | `logs\ar_tide_iter\configs\iter30.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter30\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter30.log` |
| Teacher ordered | 634/634 (1.0000) |
| Free-run ordered | **612/634 (0.9653)** |
| Decode steps | 636 |
| Eval wall (s) | 104.86 |

### iter31 · attempt 1 (2026-06-29T02:53:07)

**Hypothesis:** Resume iter17 (614); polish + in-loop AR decode ckpt (overnight)

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Config snapshot | `logs\ar_tide_iter\configs\iter31.json` (in-loop recipe; lost when registry updated) |
| Registry | `experiments.json` · `iter31` |
| Model | `models_wsl\ar\tide_overfit_iter\iter31\ar_onset_model.keras` |
| Train exit | 15 |
| Train log | `logs\ar_tide_iter\train_logs\iter31.log` |
| Error | checkpoint missing after train |

### iter31 · attempt 2 (2026-06-29T02:57:18)

**Hypothesis:** Resume iter17 (614); polish offline ckpt (overnight)

| | |
|--|--|
| Kind | retry — recipe changed in experiments.json (removed in-loop AR decode) |
| Attempt | 2 |
| Config snapshot | `logs\ar_tide_iter\configs\iter31.json` (offline recipe; overwrote attempt 1) |
| Registry | `experiments.json` · `iter31` |
| Model | `models_wsl\ar\tide_overfit_iter\iter31\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter31.log` (overwrote attempt 1) |
| Teacher ordered | 633/634 (0.9984) |
| Free-run ordered | **610/634 (0.9621)** |
| Decode steps | 636 |
| Eval wall (s) | 106.26 |

### iter32 (2026-06-29T03:10:39)

**Hypothesis:** Resume iter17; λ_inc=0.25 max_steps=48; offline ckpt

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter32`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter32.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter32\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter32.log` |
| Teacher ordered | 633/634 (0.9984) |
| Free-run ordered | **611/634 (0.9637)** |
| Decode steps | 636 |
| Eval wall (s) | 100.41 |

### iter33 (2026-06-29T03:14:24)

**Hypothesis:** Resume iter17; mild SS p=0.15 + λ_inc=0.15; offline ckpt

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter33`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter33.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter33\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter33.log` |
| Teacher ordered | 634/634 (1.0000) |
| Free-run ordered | **614/634 (0.9685)** |
| Decode steps | 636 |
| Eval wall (s) | 90.63 |

### iter34 (2026-06-29T03:29:57)

**Hypothesis:** Resume iter17; λ_residual=25; offline ckpt

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter34`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter34.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter34\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter34.log` |
| Error | teacher metrics not perfect (ordered=634/634 (1.0000), event_f1=0.9984); skipped free-run eval |

### iter35 (2026-06-29T03:32:16)

**Hypothesis:** Resume iter17; λ_inc=0.35 max_steps=64; offline ckpt

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter35`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter35.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter35\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter35.log` |
| Error | teacher metrics not perfect (ordered=633/634 (0.9984), event_f1=0.9937); skipped free-run eval |

### iter36 (2026-06-29T03:35:59)

**Hypothesis:** Resume iter21 (614); offline ckpt polish

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter36`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter36.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter36\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter36.log` |
| Teacher ordered | 634/634 (1.0000) |
| Free-run ordered | **614/634 (0.9685)** |
| Decode steps | 636 |
| Eval wall (s) | 93.08 |

### iter37 (2026-06-29T03:38:17)

**Hypothesis:** Resume iter18 (614); λ_inc=0.2; offline ckpt

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter37`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter37.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter37\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter37.log` |
| Error | teacher metrics not perfect (ordered=634/634 (1.0000), event_f1=0.9984); skipped free-run eval |

### iter38 (2026-06-29T03:41:04)

**Hypothesis:** Resume iter17; lr=1e-5 micro-polish; offline ckpt

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter38`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter38.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter38\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter38.log` |
| Error | teacher metrics not perfect (ordered=633/634 (0.9984), event_f1=0.9953); skipped free-run eval |

### iter39 (2026-06-29T03:51:28)

**Hypothesis:** Adaptive: incremental_consistency_max_steps 32 -> 16; warm-start iter36; lambda_inc=0.15 steps=16.0 lambda_res=10.0 SS=0.2 lr=2e-05

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter39`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter39.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter39\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter39.log` |
| Teacher ordered | 634/634 (1.0000) |
| Free-run ordered | **612/634 (0.9653)** |
| Decode steps | 636 |
| Eval wall (s) | 99.29 |

### iter40 (2026-06-29T03:56:13)

**Hypothesis:** Adaptive: incremental_consistency_max_steps 32 -> 48; warm-start iter36; lambda_inc=0.15 steps=48.0 lambda_res=10.0 SS=0.2 lr=2e-05

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter40`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter40.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter40\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter40.log` |
| Teacher ordered | 634/634 (1.0000) |
| Free-run ordered | **612/634 (0.9653)** |
| Decode steps | 636 |
| Eval wall (s) | 85.46 |

### iter41 (2026-06-29T03:59:53)

**Hypothesis:** Adaptive: lambda_incremental_consistency 0.15 -> 0.1; warm-start iter36; lambda_inc=0.1 steps=32 lambda_res=10.0 SS=0.2 lr=2e-05

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter41`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter41.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter41\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter41.log` |
| Teacher ordered | 634/634 (1.0000) |
| Free-run ordered | **614/634 (0.9685)** |
| Decode steps | 636 |
| Eval wall (s) | 93.08 |

### iter42 (2026-06-29T04:01:54)

**Hypothesis:** Adaptive: incremental_consistency_max_steps 32 -> 16; warm-start iter41; lambda_inc=0.1 steps=16.0 lambda_res=10.0 SS=0.2 lr=2e-05

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter42`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter42.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter42\ar_onset_model.keras` |
| Train exit | 15 |
| Train log | `logs\ar_tide_iter\train_logs\iter42.log` |
| Error | checkpoint missing after train |

### iter43 (2026-06-29T04:02:12)

**Hypothesis:** Scratch train; iter36-class recipe (lambda_inc=0.15, SS=0.2, 32 inc steps)

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter43`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter43.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter43\ar_onset_model.keras` |
| Train exit | 1 |
| Train log | `logs\ar_tide_iter\train_logs\iter43.log` |
| Error | checkpoint missing after train |

### iter43 (2026-06-29T04:02:38)

**Hypothesis:** Adaptive: incremental_consistency_max_steps 32 -> 16; warm-start iter41; lambda_inc=0.1 steps=16.0 lambda_res=10.0 SS=0.2 lr=2e-05

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter43`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter43.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter43\ar_onset_model.keras` |
| Train exit | 9 |
| Train log | `logs\ar_tide_iter\train_logs\iter43.log` |
| Error | checkpoint missing after train |

### iter43 · attempt 3 (2026-06-29T04:12:24)

**Hypothesis:** 

| | |
|--|--|
| Kind | retry — restart after killing duplicate GPU jobs |
| Attempt | 3 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter43`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter43.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter43\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter43.attempt3.log` |
| Error | teacher metrics not perfect (ordered=56/634 (0.0883), event_f1=0.1640); skipped free-run eval |

### iter44 (2026-06-29T04:20:28)

**Hypothesis:** Scratch memorization: champion lr, no SS, full 200ep

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter44`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter44.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter44\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter44.log` |
| Error | teacher metrics not perfect (ordered=257/634 (0.4054), event_f1=0.4385); skipped free-run eval |

### iter45 (2026-06-29T04:29:09)

**Hypothesis:** Scratch memorization: lr=0.0001, no SS

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter45`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter45.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter45\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter45.log` |
| Error | teacher metrics not perfect (ordered=492/634 (0.7760), event_f1=0.7571); skipped free-run eval |

### iter48 (2026-06-29T04:33:05)

**Hypothesis:** Scratch search from iter45: ar_decode_val_every_n_epochs None->10, learning_rate 5e-05->0.0001, scheduled_sampling_ramp_epochs None->80, scheduled_sampling_warmup_epochs None->50

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter48`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter48.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter48\ar_onset_model.keras` |
| Train exit | 15 |
| Train log | `logs\ar_tide_iter\train_logs\iter48.log` |
| Error | checkpoint missing after train |

### iter50 (2026-06-29T04:36:18)

**Hypothesis:** Scratch search from iter45: checkpoint_metric 'val_overfit_gate'->'val_ar_decode_ordered_onset_match', learning_rate 5e-05->0.0001, scheduled_sampling_ramp_epochs None->80, scheduled_sampling_warmup_epochs None->50

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter50`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter50.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter50\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter50.log` |
| Error | teacher metrics not perfect (ordered=492/634 (0.7760), event_f1=0.7571); skipped free-run eval |

### iter51 (2026-06-29T04:38:53)

**Hypothesis:** Scratch search from iter50: checkpoint_metric 'val_overfit_gate'->'val_ar_decode_ordered_onset_match', epochs 200->150, learning_rate 5e-05->0.0001, scheduled_sampling_ramp_epochs None->80, +1 more

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter51`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter51.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter51\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter51.log` |
| Error | teacher metrics not perfect (ordered=357/634 (0.5631), event_f1=0.5599); skipped free-run eval |

### iter173 · attempt 2 (2026-06-29T21:48:09)

**Hypothesis:** iter169 + lambda_time=3 lambda_residual=35 (fix step-318 residual)

| | |
|--|--|
| Kind | retry |
| Attempt | 2 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter173`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter173.attempt2.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter173\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter173.attempt2.log` |
| Error | teacher metrics not perfect (ordered=632/634 (0.9968), event_f1=0.9953); skipped free-run eval |

### iter174 (2026-06-30T02:12:08)

**Hypothesis:** iter169 best replay: d_model=384 lr=1e-4 iter82 losses 400ep (633/634 baseline)

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter174`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter174.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter174\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter174.log` |
| Error | teacher metrics not perfect (ordered=633/634 (0.9984), event_f1=0.9921); skipped free-run eval |

### iter175 (2026-06-30T02:22:07)

**Hypothesis:** iter169 + lambda_residual=30 (fix step-318 without iter173 time blowup)

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter175`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter175.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter175\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter175.log` |
| Error | teacher metrics not perfect (ordered=634/634 (1.0000), event_f1=0.9984); skipped free-run eval |

### iter176 (2026-06-30T02:30:04)

**Hypothesis:** Faster memorize: d_model=384 lr=1.5e-4 iter82 losses 350ep

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter176`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter176.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter176\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter176.log` |
| Error | teacher metrics not perfect (ordered=633/634 (0.9984), event_f1=0.9984); skipped free-run eval |

### iter177 (2026-06-30T02:38:23)

**Hypothesis:** Speed/capacity: d_model=320 lr=1e-4 400ep

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter177`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter177.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter177\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter177.log` |
| Error | teacher metrics not perfect (ordered=632/634 (0.9968), event_f1=0.9953); skipped free-run eval |

### iter178 (2026-06-30T02:46:42)

**Hypothesis:** iter169 + lambda_time=1.5 lambda_residual=28

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter178`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter178.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter178\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter178.log` |
| Error | teacher metrics not perfect (ordered=634/634 (1.0000), event_f1=0.9984); skipped free-run eval |

### iter179 (2026-06-30T02:48:04)

**Hypothesis:** Tier B depth: d_model=384 5-layer enc/dec lr=1e-4

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter179`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter179.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter179\ar_onset_model.keras` |
| Train exit | 1 |
| Train log | `logs\ar_tide_iter\train_logs\iter179.log` |
| Error | checkpoint missing after train |

### iter180 (2026-06-30T02:57:11)

**Hypothesis:** Tier B: d_model=448 lr=8e-5

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter180`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter180.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter180\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter180.log` |
| Error | teacher metrics not perfect (ordered=630/634 (0.9937), event_f1=0.9921); skipped free-run eval |

### iter181 (2026-06-30T03:04:55)

**Hypothesis:** Decode prep: d_model=384 mild SS p=0.1 + lambda_inc=0.15

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter181`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter181.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter181\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter181.log` |
| Error | teacher metrics not perfect (ordered=624/634 (0.9842), event_f1=0.9748); skipped free-run eval |

### iter182 (2026-06-30T03:13:04)

**Hypothesis:** Speed: d_model=384 early stop min_score=0.999 patience=15

| | |
|--|--|
| Kind | fresh |
| Attempt | 1 |
| Registry | `scripts/ar_tide_iter/experiments.json` (`iter182`) |
| Config snapshot | `logs\ar_tide_iter\configs\iter182.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter182\ar_onset_model.keras` |
| Train exit | 0 |
| Train log | `logs\ar_tide_iter\train_logs\iter182.log` |
| Error | teacher metrics not perfect (ordered=633/634 (0.9984), event_f1=0.9921); skipped free-run eval |
