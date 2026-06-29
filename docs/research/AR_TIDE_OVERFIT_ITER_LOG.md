# AR tide overfit iteration log (agent session)

Goal: free-run ordered **634/634 @ 20 ms**. Machine logs: `logs/ar_tide_iter/` (gitignored).

Started: 2026-06-28T19:37:03 · **Last updated:** 2026-06-28T22:15 (session in progress)

## Session summary (as of iter27)

|                          |                                                               |
| ------------------------ | ------------------------------------------------------------- |
| **Best free-run**        | **614/634 (96.85%)** — `iter17`, `iter18`, `iter21` (tied)    |
| **Best checkpoint path** | `models_wsl/ar/tide_overfit_iter/iter17/ar_onset_model.keras` |
| **Champion baseline**    | 611/634 (teacher 633/634)                                     |
| **Experiments run**      | 27 configs (iter07/iter16/iter24 skipped or pending)          |
| **Goal**                 | Not reached — 20 onsets still miss @ 20 ms tolerance          |

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
- Harness: `scripts/ar_tide_iter/run_exp.py`, `scripts/ar_tide_iter/build_configs.py` (outputs under `logs/ar_tide_iter/`).

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

### iter29 (2026-06-29T02:19:16)

**Hypothesis:** iter17 recipe retry post trainer fix (AR decode ckpt)

| | |
|--|--|
| Config | `logs\ar_tide_iter\configs\iter29.json` |
| Model | `models_wsl\ar\tide_overfit_iter\iter29\ar_onset_model.keras` |
| Train exit | 15 |
| Train log | `logs\ar_tide_iter\train_logs\iter29.log` |
| Error | checkpoint missing after train |

### iter29 (2026-06-29T02:28:26)

**Hypothesis:** iter17 recipe retry post trainer fix (AR decode ckpt)

| | |
|--|--|
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
