---
description: Visible console, TensorBoard stage log trees, rerun path hygiene, and serial GPU lock for scripts/
globs: scripts/**
trigger: glob
---

# Script execution — console and logs

Applies when running or editing files under `scripts/`. See [workspace.md](workspace.md) for Python executable, `_tmp/`, and `logs/` conventions.

## Visible console

- Stream train/eval/long jobs to a **visible terminal**; log-only redirect as the sole sink is not OK.
- **Console + `logs/<topic>.log` (tee)** is fine — terminal remains primary. Never launch a long job with `> logs/... 2>&1` or `*> logs/...` hiding live output.

From PowerShell or bash:

```powershell
& venv\Scripts\python.exe scripts/<script>.py <args> 2>&1 | Tee-Object -FilePath logs/<topic>_<run>.log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
```

```bash
python scripts/<script>.py <args> 2>&1 | tee logs/<topic>_<run>.log
```

**PowerShell anti-pattern (hides console):** Never combine `1> logs/out.json 2>&1 | Tee-Object ...` or `> logs/out.log 2>&1 | Tee-Object ...` — stdout binds to the file before the pipeline, starving the terminal. If a tool backgrounds a command after timeout, keep the tee pipeline attached.

## TensorBoard with training (local runs)

When launching a local **training** job that writes TensorBoard logs (`callback_root_dir` / `callbacks/.../logs`):

1. **Start TensorBoard in a separate terminal before or with the train command** — do not wait for the user to ask. (On remote cloud VMs or Colab runs, follow `workspace.md` Cloud VM defaults instead — never link `localhost` from a remote cloud VM.)
2. Point `--logdir` at the **stage** log tree, not one run folder, and tell the user the local URL (`http://localhost:6006/` by default).

| Stage | `callback_root_dir` | `--logdir` |
| ----- | ------------------- | ---------- |
| DDC placement | `callbacks/ddc/placement` | `callbacks/ddc/placement/logs` |
| DDCL placement | `callbacks/ddc/ddcl_placement` | `callbacks/ddc/ddcl_placement/logs` |
| ITGPT placement | `callbacks/ddc/itgpt_placement` | `callbacks/ddc/itgpt_placement/logs` |

Log runs to `logs/{YYYYMMDD-HHMMSS}-{run}/` rather than overwriting a shared `logs/train` root so TensorBoard overlays curves cleanly. Reuse a server on the same port when its `--logdir` already covers the new run.

## Rerun path hygiene

When the user says **rerun**, **increase epochs**, or tweak hyperparams on an existing config, keep the same `model_output_dir`, `callback_root_dir`, and log basename — do not invent `_epN` / `_run2` suffixes unless preserving a prior checkpoint is requested.

## GPU scheduling & PowerShell note

- WSL GPU jobs are **serial** (`logs/gpu_wsl.lock` via `stepcovnet.wsl_gpu_lock`) — see [wsl-gpu-stepcovnet](../skills/wsl-gpu-stepcovnet/SKILL.md).
- PowerShell: `exit_code: 1` from WSL stderr warnings is not failure by itself — read terminal or `logs/` output.
