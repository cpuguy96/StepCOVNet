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

**Gate:** Do not push Python changes unless CI would pass. Never use `--no-verify` to bypass hooks unless you explicitly ask.

| When | Command |
| ---- | ------- |
| **Every commit** (hook) | Ruff on staged files via `pre-commit` |
| **Quick local check** | `venv\Scripts\python.exe pre_submit.py --fast` (ruff only, ~seconds) |
| **Before push / PR** | `venv\Scripts\python.exe pre_submit.py --skip-install` (full CI mirror, ~30+ min) |

`pre_submit.py --fast` is the same as `--skip-install --skip-tests --skip-nbmake`. Full `pre_submit.py` mirrors [`.github/workflows/pre-submit.yml`](.github/workflows/pre-submit.yml).

**When adding or changing CI/pre-submit tooling**, fix existing violations that tooling enforces *in the same change set* — do not land hooks or workflows on a still-red tree.

Optional hooks (once): `pre-commit install --install-hooks --hook-type pre-commit` — ruff on commit only. Pre-push full-suite hook stays disabled (too slow); run full `pre_submit.py` manually before push when it matters.

**WSL GPU:** `source scripts/wsl_gpu_env.sh` after `bash scripts/wsl_ensure_env.sh` — see tracked `scripts/wsl_*.sh`.

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
