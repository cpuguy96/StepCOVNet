---
description: Universal workspace rules — Python env, portable paths, git/pre-push hygiene, and Cloud VM/Colab defaults
trigger: always_on
---

# Workspace rules

Universal rules for every session. File-specific rules (`**/*.py`, `{src,scripts,tests}/**/*`, `scripts/**`, `docs/**`) load automatically via glob triggers; task playbooks auto-discover from `.agents/skills/`.

## Python & GPU environments

| Environment | Executable & usage | Notes |
| ----------- | ------------------ | ----- |
| **Native Windows CPU** (pytest, lint, CPU scripts) | `venv\Scripts\python.exe` from **repo root** | Never use bare `python`, `pytest`, or `pip` from PATH. |
| **Windows → WSL CUDA** | `venv\Scripts\python.exe scripts/<script>.py` | Scripts with `wsl_gpu` auto-dispatch to WSL and translate the current clone path. Never set `STEPCOVNET_NO_WSL=1` for training, decode, or MERT (silently forces CPU). |
| **Direct WSL CUDA** | `"${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}"` | Run `bash scripts/wsl_ensure_env.sh` + `source scripts/wsl_gpu_env.sh` and `export STEPCOVNET_IN_WSL=1`. Do not nest another `wsl` call. |
| **Linux / Cloud VM** | Activated project venv (or bootstrapped Python) | Run from repo root. |

- **Serial WSL GPU:** Only one WSL GPU job runs at a time, enforced in code via `logs/gpu_wsl.lock` (`stepcovnet.wsl_gpu_lock`).

## Portable paths & scratch hygiene

- **Never commit machine-specific paths** (absolute Windows or WSL user-home / drive-mount paths) in tracked files; use repo-relative paths (`data/final_data`, `models_wsl/`, `callbacks/`).
- **Temp files:** Put disposable scratch under `_tmp/<topic>/` (delete when done) and command captures under `logs/<topic>_<run>.log`. Never create `tmp_*` or `_tmp_*` files at repo root or commit scratch artifacts.
- **Refresh state before linkage edits:** Before editing `AGENTS.md`, rules, skills, CI configs, or docs cross-links, check `git status`, `git ls-files`, and verify every linked path exists on disk.

## Git & pre-push hygiene

- **CI gate:** Never push Python changes unless `pre_submit.py` passes (`--fast` for ruff + pyright; `--skip-install` for full CI mirror). Never use `--no-verify` unless explicitly requested.
- **Keep CI green:** When adding or modifying CI/lint/audit tooling, fix existing violations in the same change set.
- **Clean commits:** Review `git status` / `git diff` before committing; exclude unrelated local files (e.g. `logs/`, untracked scratch) and keep docs synced with code.
- **Multi-phase tasks:** On approved multi-phase work, commit and push to the remote after **each verified phase** rather than holding commits locally until the end.
- **Windows PowerShell git:** Pass multi-line commit messages with repeated `-m` flags (not bash heredocs or scratch files). Check `git status` and commit/stash before any checkout or line-ending normalization.

## Cloud VM & Colab hard defaults

- **Git author:** Fresh cloud VMs default `/etc/gitconfig` to `noreply@github.com` (`web-flow`). Ensure `~/.gitconfig` is set to the repo owner before the first commit (`bash scripts/colab_cloud_bootstrap.sh` sets it automatically).
- **GPU priority & autonomous launch:** **`L4 → T4 → local (RTX 2080) → A100`**. On fresh cloud VMs, run `bash scripts/colab_cloud_bootstrap.sh` and launch Colab runs directly via `scripts/colab_run_remote.sh` — never push WIP code to GitHub for Colab, never hand commands back, and never claim cloud VMs cannot launch Colab. See [colab-gpu-stepcovnet](../skills/colab-gpu-stepcovnet/SKILL.md).
- **Live TensorBoard:** `scripts/colab_run_remote.sh` mirrors live `tfevents` every ~60 s to `MyDrive/StepCOVNet/live_tb/<run_name>/`. On **remote cloud VMs**, never link `http://localhost:6006` / `6008`, never use `cloudflared`, and never tell the user to run Colab cells. On **local WSL**, launch TensorBoard on port `6008+` pointing at `experiments/colab/`.
- **Long runs, VM cleanup & MERT caches:** Pass `--max-poll-seconds 420` and re-attach with `--attach`; verify `colab --auth=oauth2 sessions` shows 0 active VMs on exit. Keep MERT caches saved as `float32` (never `float16`).

## Steering corrections

On any user correction to agent process or a brain-audit request, follow [agent-maintenance](../skills/agent-maintenance/SKILL.md) in the same turn.
