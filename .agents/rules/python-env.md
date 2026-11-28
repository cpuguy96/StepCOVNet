---
description: Python executable selection (Windows venv vs WSL CUDA vs Linux/Cloud VM), portable repo-relative paths, and _tmp/logs scratch hygiene. Apply when running Python, pytest, lint, scripts, or creating temp/log files.
trigger: model_decision
---

# Python environment and portable paths

## Python & GPU executables

| Environment | Executable & usage | Notes |
| ----------- | ------------------ | ----- |
| **Native Windows CPU** (`pytest`, lint, CPU scripts) | `venv\Scripts\python.exe` from **repo root** | Never use bare `python`, `pytest`, or `pip` from PATH. |
| **Windows → WSL CUDA** | `venv\Scripts\python.exe scripts/<script>.py` | Scripts with `wsl_gpu` auto-dispatch to WSL and translate the current clone path. Never set `STEPCOVNET_NO_WSL=1` for training, eval, or MERT (silently forces CPU). |
| **Direct WSL CUDA** | `"${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}"` | Run `bash scripts/wsl_ensure_env.sh` + `source scripts/wsl_gpu_env.sh` and `export STEPCOVNET_IN_WSL=1`. Do not nest another `wsl` call. |
| **Linux / Cloud VM** | Activated project venv (or bootstrapped Python) | Run from repo root. |

## Portable paths & scratch hygiene

- **Never commit machine-specific paths** (absolute Windows or WSL user-home / drive-mount paths) in tracked files; use repo-relative paths (`data/final_data`, `models_wsl/`, `callbacks/`).
- **Temp & capture paths:** Put disposable scratch under `_tmp/<topic>/` (delete when done) and command captures under `logs/<topic>_<run>.log`. Never create `tmp_*` or `_tmp_*` files at repo root or commit scratch artifacts.
