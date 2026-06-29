# AR onset configs

Configs for autoregressive onset gates on tide (`data/v2/test/tide`). Gate **slugs** (kebab-case) name milestones in [AR_ONSET_DESIGN.md](../../docs/research/AR_ONSET_DESIGN.md); artifact directories use the same slug with underscores.

## Naming pattern

| Kind                | Pattern                                   | Example                               |
| ------------------- | ----------------------------------------- | ------------------------------------- |
| Config              | `configs/ar/<recipe>.json` or nested dir  | `configs/ar/tide.json`                |
| Checkpoint dir      | `models_wsl/ar/<gate_slug>/`              | `models_wsl/ar/gate_tide_overfit/`    |
| Decode variant      | `models_wsl/ar/gate_ar_decode/<variant>/` | `models_wsl/ar/gate_ar_decode/v2/`    |
| Perfect-overfit run | `models_wsl/ar/perfect_overfit/runN/`     | `models_wsl/ar/perfect_overfit/run5/` |
| Callbacks           | `callbacks/ar/<same path as model dir>`   | `callbacks/ar/gate_tide_overfit/`     |

Checkpoint file is always `ar_onset_model.keras` inside the model dir.

## Gate slug → config → artifacts

| Gate slug                 | Config                                                               | Model dir (canonical)                       |
| ------------------------- | -------------------------------------------------------------------- | ------------------------------------------- |
| **`gate-tide-overfit`**   | [`tide.json`](tide.json)                                             | `models_wsl/ar/gate_tide_overfit/`          |
| **`gate-ar-decode`** v2   | [`decode/v2.json`](decode/v2.json)                                   | `models_wsl/ar/gate_ar_decode/v2/`          |
| **perfect-overfit** run 1 | [`overfit_perfect/base.json`](overfit_perfect/base.json)             | `models_wsl/ar/perfect_overfit/run1/`       |
| perfect-overfit run 2–5   | [`overfit_perfect/runN.json`](overfit_perfect/)                      | `models_wsl/ar/perfect_overfit/runN/`       |
| perfect-overfit smoke     | [`overfit_perfect/run4_smoke.json`](overfit_perfect/run4_smoke.json) | `models_wsl/ar/perfect_overfit/run4_smoke/` |

Decode sketches (warm-start only, no default output dir): [`decode/tide.json`](decode/tide.json), [`decode/perfect.json`](decode/perfect.json).

## Informal aliases (logs, chat, EXP text)

| Informal              | Means                                                    | Canonical checkpoint                                           |
| --------------------- | -------------------------------------------------------- | -------------------------------------------------------------- |
| **`gate_v5`**         | 5th training attempt that **passed** `gate-tide-overfit` | `models_wsl/ar/gate_tide_overfit/ar_onset_model.keras`         |
| `gate_v2` … `gate_v4` | Earlier **failed** tide-overfit attempts (historical)    | see [EXPERIMENT_LOG.md](../../docs/research/EXPERIMENT_LOG.md) |
| `perfect_v5`          | Shorthand for perfect-overfit **run 5**                  | `models_wsl/ar/perfect_overfit/run5/`                          |

## Migration from old `models_wsl/` paths (2026-06)

Historical runs wrote flat names like `ar_tide_overfit_gate_v5`. New training uses the table above. **Do not rename checkpoints in git** (artifacts are local / gitignored). Copy or symlink if you still have old dirs:

```powershell
# from repo root — example: gate-tide-overfit PASS checkpoint
New-Item -ItemType Directory -Force models_wsl\ar\gate_tide_overfit
Copy-Item models_wsl\ar_tide_overfit_gate_v5\ar_onset_model.keras models_wsl\ar\gate_tide_overfit\
```

| Old path                                       | New path                                    |
| ---------------------------------------------- | ------------------------------------------- |
| `models_wsl/ar_tide_overfit_gate_v5/`          | `models_wsl/ar/gate_tide_overfit/`          |
| `models_wsl/ar_tide_overfit_gate_decode_v2/`   | `models_wsl/ar/gate_ar_decode/v2/`          |
| `models_wsl/ar_tide_overfit_perfect/`          | `models_wsl/ar/perfect_overfit/run1/`       |
| `models_wsl/ar_tide_overfit_perfect_v2/`       | `models_wsl/ar/perfect_overfit/run2/`       |
| `models_wsl/ar_tide_overfit_perfect_v3/`       | `models_wsl/ar/perfect_overfit/run3/`       |
| `models_wsl/ar_tide_overfit_perfect_v4/`       | `models_wsl/ar/perfect_overfit/run4/`       |
| `models_wsl/ar_tide_overfit_perfect_v5/`       | `models_wsl/ar/perfect_overfit/run5/`       |
| `models_wsl/ar_tide_overfit_perfect_v4_smoke/` | `models_wsl/ar/perfect_overfit/run4_smoke/` |

[EXPERIMENT_LOG.md](../../docs/research/EXPERIMENT_LOG.md) entries keep old paths as written at run time; use this table when resuming work.
