---
description: Which Python executable to use — Windows venv (CPU) vs WSL (GPU)
alwaysApply: true
---

# Python environment

## Windows development model

StepCOVNet on Windows uses **two environments**:

| Layer | Where | Purpose |
| ----- | ----- | ------- |
| **Windows venv** | `venv\` beside the clone | pytest, lint, CPU scripts — any clone path |
| **WSL Linux venv** | default `$HOME/stepcovnet-venv-wsl` in WSL | TensorFlow/PyTorch CUDA — not tied to a Windows username path |

GPU scripts that call `wsl_gpu.maybe_dispatch_for_training` re-exec in WSL and translate the **current clone path** automatically. Do not hardcode `/mnt/c/Users/...` or `C:\Users\...`.

Override WSL venv location: `WSL_VENV` or `STEPCOVNET_WSL_PYTHON`. Opt out of WSL dispatch: `STEPCOVNET_NO_WSL=1`.

`STEPCOVNET_NO_WSL=1` **silently forces Windows CPU** — it does not fall back to GPU. Never set it for training, decode, or MERT extraction (measured ~3.5× slower on an AR decode); reserve it for CPU-only debugging where you intend to skip CUDA.

## Decision

| Workload                                            | Where          | Executable                                                    |
| --------------------------------------------------- | -------------- | ------------------------------------------------------------- |
| pytest, lint, edits, CPU scripts                    | Native Windows | `venv\Scripts\python.exe` (from **repo root**)                |
| Training, MERT `--device=cuda`, any CUDA on Windows | WSL            | `"${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}"` |

Do not use bare `python`, `pytest`, or `pip` from PATH.

## Windows venv (CPU)

Run from **repository root**:

```text
venv\Scripts\python.exe -m pytest ...
venv\Scripts\python.exe -m pip ...
```

## WSL GPU (summary)

Full command templates: [wsl-gpu-stepcovnet skill](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md).

Shared shell vars: `scripts/wsl_common.sh`. Bootstrap: `scripts/wsl_ensure_env.sh`. Always `source scripts/wsl_gpu_env.sh` before TensorFlow import. Set `STEPCOVNET_IN_WSL=1`. If already inside WSL, do not nest another `wsl` call.

**Serial GPU:** Only one WSL GPU job at a time. Enforced in code via `logs/gpu_wsl.lock` (`stepcovnet.wsl_gpu_lock`) on dispatch and WSL-direct training/decode/MERT — not skill-only.
