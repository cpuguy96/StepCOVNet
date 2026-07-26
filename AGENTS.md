# StepCOVNet — agent entry

Open **one** index below. Do not load multiple. Specialized rows (e.g. dataset prep, what's next) are **alternatives** to the general research/skills index — not extra docs to load together.

| If the task is… | Index |
| --------------- | ----- |
| A **procedure** (train, overfit, WSL, log EXP/NOTE, debug playbooks) | [.cursor/skills/README.md](.cursor/skills/README.md) |
| **Research** (experiments, design, pipeline; log runs in EXPERIMENT_LOG) | [docs/research/README.md](docs/research/README.md) |
| **Onset metrics** (`timing_match`, F1, gates) | [docs/research/ONSET_METRICS.md](docs/research/ONSET_METRICS.md) |
| **AR onset design** (locked stack, gates, debug notes) | [docs/research/AR_ONSET_DESIGN.md](docs/research/AR_ONSET_DESIGN.md) |
| **Scaling AR training** (how to add songs and keep numbers comparable) | [docs/research/AR_SCALING_LADDER.md](docs/research/AR_SCALING_LADDER.md) |
| **What's next** (current phase) | [docs/research/EXPERIMENT_LOG.md](docs/research/EXPERIMENT_LOG.md) § Current phase |
| **Research autoresearch** (one-prompt goal + budget loop) | [.cursor/skills/autoresearch/SKILL.md](.cursor/skills/autoresearch/SKILL.md) |
| **AR tide overfit** (gate **PASS** — champion v8) | [EXPERIMENT_LOG.md](docs/research/EXPERIMENT_LOG.md) § Current phase · champion: [configs/ar/tide_overfit.json](configs/ar/tide_overfit.json) |
| GPU training, WSL, overfit smoke tests | [.cursor/skills/wsl-gpu-stepcovnet/SKILL.md](.cursor/skills/wsl-gpu-stepcovnet/SKILL.md) |
| Repo layout (`src/`, `scripts/`, `configs/`, data) | [docs/agents/project-layout.md](docs/agents/project-layout.md) |
| Dataset prep / `final_data` pipeline | [docs/research/DATASET_PREP_PIPELINE.md](docs/research/DATASET_PREP_PIPELINE.md) |
| **Recreate training data** (SMO downloads → prep) | [docs/research/TRAINING_DATA_SETUP.md](docs/research/TRAINING_DATA_SETUP.md) |
| Pre-push validation (same as CI) | [pre_submit.py](pre_submit.py) · [`.github/workflows/pre-submit.yml`](.github/workflows/pre-submit.yml) |
| Steering correction — how the agent decides / promotes lessons | [.cursor/skills/steering-correction-promotion/SKILL.md](.cursor/skills/steering-correction-promotion/SKILL.md) |
| **Refresh agent brain** / **refresh all rules** — audit rules, skills, catalogs | [.cursor/skills/agent-brain-refresh/SKILL.md](.cursor/skills/agent-brain-refresh/SKILL.md) |

North star: [PIPELINE_ARCHITECTURE.md](docs/research/PIPELINE_ARCHITECTURE.md).

**Rules catalog** (always vs scoped — Cursor loads `alwaysApply` automatically): [docs/agents/agent-brain.md](docs/agents/agent-brain.md).

---

## Before commit / push

**Gate:** Do not push Python changes unless CI would pass. Never use `--no-verify` to bypass hooks unless you explicitly ask.

| When | Command |
| ---- | ------- |
| **Every commit** (hook) | Ruff on staged files via `pre-commit` |
| **Quick local check** | `python pre_submit.py --fast` (ruff only, ~seconds) |
| **Before push / PR** | `python pre_submit.py --skip-install` (full CI mirror, ~30+ min) |

`pre_submit.py --fast` is the same as `--skip-install --skip-tests --skip-nbmake`. Full `pre_submit.py` mirrors [`.github/workflows/pre-submit.yml`](.github/workflows/pre-submit.yml).

**Agent commits:** Clean up the change set before `git commit` — review `git status` / `git diff`, drop unrelated or local-only files (e.g. iter `experiments.json`, `AR_TIDE_OVERFIT_ITER_LOG.md`, `logs/` unless explicitly requested), fix lint in staged code, keep docs in sync with the harness diff. Do not commit a messy working tree.

**When adding or changing CI/pre-submit tooling**, fix existing violations that tooling enforces *in the same change set* — do not land hooks or workflows on a still-red tree.

Optional hooks (once): `pre-commit install --install-hooks --hook-type pre-commit` — ruff on commit only. Pre-push full-suite hook stays disabled (too slow); run full `pre_submit.py` manually before push when it matters.

**WSL GPU:** `source scripts/wsl_gpu_env.sh` after `bash scripts/wsl_ensure_env.sh` — see tracked `scripts/wsl_*.sh`.

---

## Path conventions (portable)

| Kind | Pattern |
| ---- | ------- |
| CPU (pytest, lint, scripts) | `python` from repository root with project venv activated |
| WSL GPU | `python` after `source scripts/wsl_gpu_env.sh` — override with `STEPCOVNET_WSL_PYTHON` or `WSL_VENV` |
| Data / artifacts | Repo-relative (`data/v2`, `data/final_data`, `models_wsl/`) |

GPU scripts on Windows auto-dispatch to WSL using the **current clone path** — do not commit machine-specific absolute paths (`C:\Users\...`, `/mnt/c/Users/...`).
