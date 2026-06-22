---
name: wsl-gpu-stepcovnet
description: Runs StepCOVNet GPU training and MERT extraction on Windows via WSL with TensorFlow CUDA libraries. Use for train_onset, overfit scripts, extract_mert_features with cuda, or when TensorFlow sees zero GPUs in WSL.
disable-model-invocation: true
---

# WSL GPU (StepCOVNet)

## Windows development model

On Windows, **CPU work stays in the clone's Windows venv**; **CUDA work runs in WSL**. The repo clone can live on any drive or folder — dispatch code derives WSL paths from the current checkout, not from a fixed username or `/mnt/c/Users/...` string.

| Setting | Default | Override |
| ------- | ------- | -------- |
| WSL venv root | `$HOME/stepcovnet-venv-wsl` | `WSL_VENV` |
| WSL Python | `$WSL_VENV/bin/python` | `STEPCOVNET_WSL_PYTHON` |
| Repo root | cwd / script location | — |

Shell helpers share these via `scripts/wsl_common.sh`.

## Rule

On native Windows, **never** run CUDA/GPU training in the Windows venv. From **repo root**:

```text
venv\Scripts\python.exe scripts/<script>.py <args>
```

when the script calls `wsl_gpu.maybe_dispatch_for_training` (or MERT `--device=cuda` dispatch). See [python-environment.mdc](../../.cursor/rules/python-environment.mdc).

Opt out: `STEPCOVNET_NO_WSL=1`.

## Manual WSL (debugging or already inside WSL)

```bash
cd /path/to/your/clone   # any mount, e.g. /mnt/d/dev/stepcovnet
bash scripts/wsl_ensure_env.sh
source scripts/wsl_gpu_env.sh
export STEPCOVNET_IN_WSL=1
"${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}" scripts/<script>.py <args>
```

Without `source wsl_gpu_env.sh`, TensorFlow may silently fall back to CPU.

## Onset event training

| Script                                  | Purpose                                              |
| --------------------------------------- | ---------------------------------------------------- |
| `scripts/train_onset_event.py`          | Single experiment from JSON config                   |
| `scripts/run_overfit_tide_suite.py`     | Tide overfit: conv1d / mel / mert (`--epochs=50`)    |
| `scripts/run_overfit_tide_ablations.py` | Threshold sweep, loss/arch variants (MERT)           |
| `scripts/run_overfit_tide_bisection.py` | Diagnose, grid oracle, half-cheat ablations (EXP-11) |
| `scripts/debug_onset_overfit.py`        | Checkpoint diagnostics                               |

**Configs:** `configs/overfit_tide/{conv1d,mel,mert}.json` (default **50 epochs**).

**Artifacts:** `models_wsl/overfit_tide/`, `models_wsl/overfit_tide_ablations/`, MERT cache `data/v2/test/tide.mert.npy`.

## Other GPU scripts

| Script                                           | Track              |
| ------------------------------------------------ | ------------------ |
| `scripts/train_onset.py`                         | Dense onset        |
| `scripts/train_arrow.py`                         | Arrow model        |
| `scripts/extract_mert_features.py --device=cuda` | MERT feature cache |

## Already inside WSL

`cd` to repo root, `bash scripts/wsl_ensure_env.sh` if needed, `source scripts/wsl_gpu_env.sh`, `export STEPCOVNET_IN_WSL=1`, use `"${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}"` — do not nest another `wsl` call.
