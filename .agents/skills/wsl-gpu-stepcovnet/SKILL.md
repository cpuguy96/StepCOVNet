---
name: wsl-gpu-stepcovnet
description: Runs StepCOVNet GPU training and MERT extraction on Windows via WSL with PyTorch CUDA libraries. Use for train_*_placement, train_onset, extract_mert_features with cuda, or when PyTorch sees zero GPUs in WSL.
---

# WSL GPU (StepCOVNet)

## Windows development model

On Windows, **CPU work stays in the clone's Windows venv**; **CUDA work runs in WSL**. The repo clone can live on any drive or folder — dispatch code derives WSL paths from the current checkout, not from a fixed username or drive-mount path. See [python-env.md](../../rules/python-env.md) for Python executable overrides (`WSL_VENV`, `STEPCOVNET_WSL_PYTHON`, `STEPCOVNET_IN_WSL`, `STEPCOVNET_NO_WSL`).

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
| **No train + infer overlap** | Do not run GPU **training** and GPU **inference** (`eval_*`, `generate.py`) in **separate processes** at the same time. |
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

Without `source wsl_gpu_env.sh`, PyTorch may silently fall back to CPU.

## Placement & onset scripts

| Script | Track |
| ------ | ----- |
| `scripts/train_itgpt_placement.py` | ITGPT hierarchical transformer placement on Dataset B |
| `scripts/eval_itgpt_placement.py` | `M-slot48` F1 @ 0.5, max-F1, grounding & error breakdown |
| `scripts/train_ddcl_placement.py` | DDCL ConvLSTM 48-slot placement on Dataset A. Full split: on-demand `window_at_beat` in `stepcovnet.ddcl.datasets`. |
| `scripts/eval_ddcl_placement.py` | `M-slot48` F1 @ 0.5 and max-F1 |
| `scripts/train_ddc_placement.py` | DDC C-LSTM placement on Dataset A |
| `scripts/eval_ddc_placement.py` | Hamming ±20 ms F-score_c / F-score_m |
| `scripts/train_onset.py` | Dense onset — use `--training_index_path` for `final_data` |
| `scripts/train_arrow.py` | Arrow model |
| `scripts/extract_mert_features.py --device=cuda` | MERT feature cache |

## Already inside WSL

`cd` to repo root, `bash scripts/wsl_ensure_env.sh` if needed, `source scripts/wsl_gpu_env.sh`, `export STEPCOVNET_IN_WSL=1`, use `python` — do not nest another `wsl` call.

## Console, TensorBoard, and rerun hygiene

- **Visible console + tee:** Stream train/eval/long jobs to a **visible terminal**; log-only redirect (`> logs/... 2>&1` or `*> logs/...`) is not OK. Capture via PowerShell or bash tee:

```powershell
& venv\Scripts\python.exe scripts/<script>.py <args> 2>&1 | Tee-Object -FilePath logs/<topic>_<run>.log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

```bash
python scripts/<script>.py <args> 2>&1 | tee logs/<topic>_<run>.log
```

  In PowerShell, never combine `1> logs/out.json 2>&1 | Tee-Object ...` (file binds stdout before Tee and starves the terminal). If a tool backgrounds a job after timeout, keep the tee pipeline attached. Note that `exit_code: 1` from WSL stderr warnings is not failure by itself — read terminal or `logs/` output.
- **TensorBoard with local training:** Start TensorBoard in a separate terminal before or with the train command, pointing `--logdir` at the **stage** tree (never a single run folder), and tell the user the local URL (`http://localhost:6006/` by default):

| Stage | `callback_root_dir` | `--logdir` |
| ----- | ------------------- | ---------- |
| DDC placement | `callbacks/ddc/placement` | `callbacks/ddc/placement/logs` |
| DDCL placement | `callbacks/ddc/ddcl_placement` | `callbacks/ddc/ddcl_placement/logs` |
| ITGPT placement | `callbacks/ddc/itgpt_placement` | `callbacks/ddc/itgpt_placement/logs` |

  Log runs to timestamped `logs/{YYYYMMDD-HHMMSS}-{run}/` folders (never overwrite a shared `logs/train` root). On remote cloud VMs or Colab runs, use [colab-gpu-stepcovnet](../colab-gpu-stepcovnet/SKILL.md) live Drive sync instead of linking `localhost`.
- **Rerun path hygiene:** When rerunning or increasing epochs on an existing config, reuse the same `model_output_dir`, `callback_root_dir`, and log basename — do not invent `_epN` / `_run2` suffixes unless preserving a prior checkpoint is requested.
