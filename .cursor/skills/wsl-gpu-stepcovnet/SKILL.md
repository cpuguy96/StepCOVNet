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

On native Windows, **never** run CUDA/GPU training in the Windows venv. From **repo root** (project venv activated):

```text
python scripts/<script>.py <args>
```

when the script calls `wsl_gpu.maybe_dispatch_for_training` (or MERT `--device=cuda` dispatch).

Opt out: `STEPCOVNET_NO_WSL=1`.

## GPU scheduling (one job at a time)

The WSL GPU is **single-tenant** for the agent:

| Rule | Detail |
| ---- | ------ |
| **One training job** | Do not start a second WSL GPU **training** run while one is already active (train, overfit, MERT extract with `--device=cuda`). |
| **No train + infer overlap** | Do not run GPU **training** and GPU **inference** (`debug_*`, `--ar_decode`, `eval_dense_*`, `model.predict`) in **separate processes** at the same time. |
| **Before launching** | `assert_wsl_gpu_free_for_training()` checks `nvidia-smi` **and** `logs/gpu_wsl.lock` (`stepcovnet.wsl_gpu_lock`). Windows dispatch acquires the lock before `wsl`; a second parallel launch fails on `O_EXCL`. Override: `STEPCOVNET_FORCE_GPU=1`. |
| **Manual check** | `python -c "from stepcovnet import wsl_gpu; wsl_gpu.assert_wsl_gpu_free_for_training()"` |
| **After training** | Offline decode/eval on GPU is fine **sequentially** once training has exited. |

CPU scripts (pytest, lint, config edits) may run while a GPU job is active — they do not use the WSL CUDA device.

## Manual WSL (debugging or already inside WSL)

```bash
cd /path/to/your/clone   # any mount, e.g. /mnt/d/dev/stepcovnet
bash scripts/wsl_ensure_env.sh
source scripts/wsl_gpu_env.sh
export STEPCOVNET_IN_WSL=1
python scripts/<script>.py <args>
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

**`final_data`:** pass `--training_index_path=data/final_data/training_index.json` (or a smoke subset manifest). Do not rely on `data_dir=val_data_dir` alone when train/val must follow the P8 song split.

**Configs:** `configs/overfit_tide/{conv1d,mel,mert}.json` (default **50 epochs**); `configs/event/audio_baseline.json` for multi-song event runs (`num_queries` / `n_max_onsets` = **2048**).

**Artifacts:** `models_wsl/overfit_tide/`, `models_wsl/overfit_tide_ablations/`, MERT cache `data/v2/test/tide.mert.npy`.

## Other GPU scripts

| Script                                           | Track              |
| ------------------------------------------------ | ------------------ |
| `scripts/train_onset.py`                         | Dense onset — use `--training_index_path` for `final_data` |
| `scripts/train_ddc_placement.py`                 | DDC C-LSTM placement on Dataset A |
| `scripts/eval_ddc_placement.py`                  | Hamming ±20 ms F-score_c / F-score_m |
| `scripts/train_ddcl_placement.py`                | DDCL ConvLSTM 48-slot placement on Dataset A |
| `scripts/eval_ddcl_placement.py`                 | `M-slot48` F1 @ 0.5 and max-F1 |
| `scripts/train_arrow.py`                         | Arrow model        |
| `scripts/extract_mert_features.py --device=cuda` | MERT feature cache |

## Already inside WSL

`cd` to repo root, `bash scripts/wsl_ensure_env.sh` if needed, `source scripts/wsl_gpu_env.sh`, `export STEPCOVNET_IN_WSL=1`, use `python` — do not nest another `wsl` call.

## Console and logs

When running scripts under `scripts/`, follow scoped rule [scripts-execution.mdc](../../rules/scripts-execution.mdc): visible terminal, **TensorBoard up before/with training**, captures under `logs/`, scratch under `_tmp/`.
