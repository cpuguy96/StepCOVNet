# StepCOVNet — agent entry

Open **one** index below. Do not load multiple.

| If the task is… | Index |
| --------------- | ----- |
| GPU training, WSL, overfit smoke tests | [.cursor/skills/wsl-gpu-stepcovnet/SKILL.md](.cursor/skills/wsl-gpu-stepcovnet/SKILL.md) (local) |
| Repo layout (`src/`, `scripts/`, `configs/`, data) | [docs/agents/project-layout.md](docs/agents/project-layout.md) (local) |
| Dataset prep / `final_data` pipeline | [docs/research/DATASET_PREP_PIPELINE.md](docs/research/DATASET_PREP_PIPELINE.md) (local) |
| Pre-push validation (same as CI) | [pre_submit.py](pre_submit.py) · [`.github/workflows/pre-submit.yml`](.github/workflows/pre-submit.yml) |

`docs/` and `.cursor/` are gitignored but may exist locally — verify paths with `git ls-files` vs on-disk before linking from **tracked** files.

---

## Always-on rules (`.cursor/rules/`, local)

| Rule file | Topic |
| --------- | ----- |
| `agents-entry.mdc` | Route through this file; one index per task |
| `state-and-paths.mdc` | Refresh repo state; no machine-specific paths in tracked files |
| `python-environment.mdc` | `venv\Scripts\python.exe` (CPU) vs WSL GPU venv |
| `python-style.mdc` | Ruff (`ruff check .`), pydoclint, pyright |
| `python-tests.mdc` | Tests + coverage for code changes |
| `research-logging.mdc` | EXP/NOTE logging under `docs/research/` |

---

## Before commit / push

From **repository root** (clone path arbitrary):

```text
venv\Scripts\python.exe pre_submit.py
```

Optional hooks (once): `pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push`

CI workflow: [`.github/workflows/pre-submit.yml`](.github/workflows/pre-submit.yml) — `ruff check .`, full `pytest`, `nbmake notebooks`.

---

## Path conventions (portable)

| Kind | Pattern |
| ---- | ------- |
| Windows CPU | `venv\Scripts\python.exe` from repo root (clone anywhere) |
| WSL GPU | `"${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}"` after `source scripts/wsl_gpu_env.sh` — Linux `$HOME`, not a Windows user path |
| WSL venv override | `WSL_VENV` or `STEPCOVNET_WSL_PYTHON` |
| Data / artifacts | Repo-relative (`data/v2`, `data/final_data`, `models_wsl/`) |

GPU scripts on Windows auto-dispatch to WSL using the **current clone path** — never embed `/mnt/c/Users/...` or `C:\Users\...`.

Scan tracked files: `git grep -E "C:\\\\Users|/mnt/c/Users/"` — expect no matches.
