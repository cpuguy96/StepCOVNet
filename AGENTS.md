# StepCOVNet — agent entry

Open **one** index below. Do not load multiple. Specialized rows (e.g. dataset prep, what's next) are **alternatives** to the general research/skills index — not extra docs to load together.

| If the task is… | Index |
| --------------- | ----- |
| A **procedure** (train, overfit, WSL, log EXP/NOTE, debug playbooks) | [.agents/skills/README.md](.agents/skills/README.md) |
| **Research** (experiments, design, pipeline; log runs in EXPERIMENT_LOG) | [docs/research/README.md](docs/research/README.md) |
| **Paper draft / citations** (related work, BibTeX keys, claim ledger) | [docs/research/PAPER_LEDGER.md](docs/research/PAPER_LEDGER.md) · [paper.bib](docs/research/paper.bib) · [PAPER_OUTLINE.md](docs/research/PAPER_OUTLINE.md) |
| **DDC placement recreation** (Dataset A, `M-ddc-20ms`) | [configs/ddc/placement_fraxtil.json](configs/ddc/placement_fraxtil.json) · `scripts/train_ddc_placement.py` |
| **DDCL / ITGPT recreation** (then incremental claims) | [PAPER_LEDGER.md](docs/research/PAPER_LEDGER.md) · [configs/ddc/ddcl_placement_fraxtil.json](configs/ddc/ddcl_placement_fraxtil.json) · [configs/ddc/itgpt_placement_fraxtil_exp.json](configs/ddc/itgpt_placement_fraxtil_exp.json) · [configs/ddc/itgpt_placement_fraxtil_exp_128ep.json](configs/ddc/itgpt_placement_fraxtil_exp_128ep.json) · [configs/ddc/itgpt_placement_fraxtil_orig_128ep.json](configs/ddc/itgpt_placement_fraxtil_orig_128ep.json) · `scripts/train_ddcl_placement.py` · `scripts/train_itgpt_placement.py` |
| **Onset metrics** (`timing_match`, F1, gates) | [docs/research/ONSET_METRICS.md](docs/research/ONSET_METRICS.md) |
| **AR onset design** (locked stack, gates, debug notes) | [docs/research/AR_ONSET_DESIGN.md](docs/research/AR_ONSET_DESIGN.md) |
| **Scaling AR training** (how to add songs and keep numbers comparable) | [docs/research/AR_SCALING_LADDER.md](docs/research/AR_SCALING_LADDER.md) |
| **What's next** / **what to do now** / orientation | [.agents/skills/whats-next/SKILL.md](.agents/skills/whats-next/SKILL.md) → [EXPERIMENT_LOG.md](docs/research/EXPERIMENT_LOG.md) § Current phase |
| **Research autoresearch** (one-prompt goal + budget loop) | [.agents/skills/autoresearch/SKILL.md](.agents/skills/autoresearch/SKILL.md) |
| GPU training, WSL, overfit smoke tests | [.agents/skills/wsl-gpu-stepcovnet/SKILL.md](.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |
| **Colab GPU training & Drive sync** (parallel cloud runs, never GitHub) | [.agents/skills/colab-gpu-stepcovnet/SKILL.md](.agents/skills/colab-gpu-stepcovnet/SKILL.md) |
| Repo layout (`src/`, `scripts/`, `configs/`, data) | [docs/agents/project-layout.md](docs/agents/project-layout.md) |
| Dataset prep / `final_data` pipeline | [docs/research/DATASET_PREP_PIPELINE.md](docs/research/DATASET_PREP_PIPELINE.md) |
| **Recreate training data** (SMO downloads → prep) | [docs/research/TRAINING_DATA_SETUP.md](docs/research/TRAINING_DATA_SETUP.md) |
| Pre-push validation (same as CI) | [pre_submit.py](pre_submit.py) · [`.github/workflows/pre-submit.yml`](.github/workflows/pre-submit.yml) |
| Steering correction — how the agent decides / promotes lessons | [.agents/skills/steering-correction-promotion/SKILL.md](.agents/skills/steering-correction-promotion/SKILL.md) |
| **Refresh agent brain** / **refresh all rules** — audit rules, skills, catalogs | [.agents/skills/agent-brain-refresh/SKILL.md](.agents/skills/agent-brain-refresh/SKILL.md) |

North star: [PIPELINE_ARCHITECTURE.md](docs/research/PIPELINE_ARCHITECTURE.md).

**Rules catalog** (always-on vs scoped — always-on rules live in `.agents/rules/`): [docs/agents/agent-brain.md](docs/agents/agent-brain.md).

---

## Before commit / push

**Gate:** Do not push Python changes unless CI would pass. Never use `--no-verify` to bypass hooks unless you explicitly ask.

| When | Command |
| ---- | ------- |
| **Every commit** (hook) | Ruff on staged files via `pre-commit` |
| **Quick local check** | `python pre_submit.py --fast` (ruff + pyright, ~seconds) |
| **Before push / PR** | `python pre_submit.py --skip-install` (full CI mirror, ~2–4 min) |

`pre_submit.py --fast` is the same as `--skip-install --skip-tests --skip-nbmake`. Full `pre_submit.py` mirrors [`.github/workflows/pre-submit.yml`](.github/workflows/pre-submit.yml).

**Agent commits:** Clean up the change set before `git commit` — review `git status` / `git diff`, drop unrelated or local-only files (e.g. iter `experiments.json`, `AR_TIDE_OVERFIT_ITER_LOG.md`, `logs/` unless explicitly requested), fix lint in staged code, keep docs in sync with the harness diff. Do not commit a messy working tree.

**When adding or changing CI/pre-submit tooling**, fix existing violations that tooling enforces *in the same change set* — do not land hooks or workflows on a still-red tree.

Optional hooks (once): `pre-commit install --install-hooks --hook-type pre-commit` — ruff on commit only. Pre-push full-suite hook stays disabled (too slow); run full `pre_submit.py` manually before push when it matters.

**WSL GPU:** `source scripts/wsl_gpu_env.sh` after `bash scripts/wsl_ensure_env.sh` — see tracked `scripts/wsl_*.sh`.

**Colab sync (proactive, private Drive only — never GitHub):** Whenever preparing experiments or checking Colab results, proactively open and follow [.agents/skills/colab-gpu-stepcovnet/SKILL.md](.agents/skills/colab-gpu-stepcovnet/SKILL.md) without waiting to be asked.

### Cloud VM & Colab hard defaults (every session)

- **Git author:** Fresh cloud VMs default `/etc/gitconfig` to `noreply@github.com` (`web-flow`). Before the first commit, ensure `~/.gitconfig` is set to the repo owner (`bash scripts/colab_cloud_bootstrap.sh` sets it automatically from repo history).
- **GPU priority & autonomous launch:** **`L4 → T4 → local (RTX 2080) → A100`**. On a fresh cloud VM, run `bash scripts/colab_cloud_bootstrap.sh` (installs `colab` on Python 3.13 + `rclone` + auto-pulls `MyDrive/StepCOVNet/.secrets/rclone.conf`) and launch directly via `scripts/colab_run_remote.sh` — never claim cloud VMs cannot launch Colab.
- **Live TensorBoard (cloud VMs vs local WSL):** `scripts/colab_run_remote.sh` mirrors live `tfevents` every ~60 s to **`MyDrive/StepCOVNet/live_tb/<run_name>/`** so the user's own PC TensorBoard (via Google Drive for Desktop) picks up cloud-VM runs live. On cloud VMs, never link `http://localhost:6006` / `6008` (hits user's own PC), never use `cloudflared`, and never tell the user to run Colab cells. On local WSL, launch local TensorBoard on port `6008+` pointing at `experiments/colab/`.
- **Long runs & VM cleanup:** Pass `--max-poll-seconds 420` and re-attach with `--attach` to stay under cloud-VM command timeouts; verify `colab --auth=oauth2 sessions` shows 0 active VMs when finished.
- **MERT caches:** Keep saved as **`float32`** (never `float16`); stream `__meta.tar` + `__mert_*.tar` archives for MERT runs.

---

## Path conventions (portable)

| Kind | Pattern |
| ---- | ------- |
| CPU (pytest, lint, scripts) | `python` from repository root with project venv activated |
| WSL GPU | `python` after `source scripts/wsl_gpu_env.sh` — override with `STEPCOVNET_WSL_PYTHON` or `WSL_VENV` |
| Data / artifacts | Repo-relative (`data/v2`, `data/final_data`, `models_wsl/`) |

GPU scripts on Windows auto-dispatch to WSL using the **current clone path** — do not commit machine-specific absolute paths (`C:\Users\...`, `/mnt/c/Users/...`).
