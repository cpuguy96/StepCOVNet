# StepCOVNet — agent entry

Open **one** relevant doc or skill below per task. Do not load multiple index files together.

| If the task is… | Start at |
| --------------- | -------- |
| **What's next / status / orientation** | [whats-next](.agents/skills/whats-next/SKILL.md) → [EXPERIMENT_LOG.md](docs/research/EXPERIMENT_LOG.md) § Current phase |
| **Research** (experiments, design, logging) | [docs/research/README.md](docs/research/README.md) · [research-session-workflow](.agents/skills/research-session-workflow/SKILL.md) |
| **Autoresearch** (goal + budget loop) | [autoresearch](.agents/skills/autoresearch/SKILL.md) |
| **Paper draft / citations** | [PAPER_LEDGER.md](docs/research/PAPER_LEDGER.md) · [paper.bib](docs/research/paper.bib) · [PAPER_OUTLINE.md](docs/research/PAPER_OUTLINE.md) |
| **DDC / DDCL / ITGPT placement** | [placement_fraxtil.json](configs/ddc/placement_fraxtil.json) · [ddcl_placement_fraxtil.json](configs/ddc/ddcl_placement_fraxtil.json) · [itgpt_placement_fraxtil_orig_128ep.json](configs/ddc/itgpt_placement_fraxtil_orig_128ep.json) |
| **Onset evaluation & metrics** | [onset-evaluation](.agents/skills/onset-evaluation/SKILL.md) · [ONSET_METRICS.md](docs/research/ONSET_METRICS.md) |
| **Dataset prep / recreating training data** | [DATASET_PREP_PIPELINE.md](docs/research/DATASET_PREP_PIPELINE.md) · [TRAINING_DATA_SETUP.md](docs/research/TRAINING_DATA_SETUP.md) |
| **GPU training — Colab (preferred) or WSL** | [colab-gpu-stepcovnet](.agents/skills/colab-gpu-stepcovnet/SKILL.md) · [wsl-gpu-stepcovnet](.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |
| **Repo layout** (`src/`, `scripts/`, `configs/`, data) | [project-layout.md](docs/agents/project-layout.md) |
| **Git, commits, CI & PR hygiene** | [git-and-ci.md](.agents/rules/git-and-ci.md) |
| **Python env, style & tests** | [python-env.md](.agents/rules/python-env.md) · [python-style.md](.agents/rules/python-style.md) · [python-tests.md](.agents/rules/python-tests.md) |
| **Research doc schemas** | [docs-and-research.md](.agents/rules/docs-and-research.md) |
| **Steering correction / brain audit / evals** | [agent-maintenance](.agents/skills/agent-maintenance/SKILL.md) · `scripts/audit_agent_brain.py` · `scripts/eval_agent_setup.py` |

- **North star:** [PIPELINE_ARCHITECTURE.md](docs/research/PIPELINE_ARCHITECTURE.md)
- **Pre-submit gate (CI mirror):** `python pre_submit.py --fast` (quick static checks) · `python pre_submit.py --skip-install` (full suite; see [pre-submit.yml](.github/workflows/pre-submit.yml))
- **Rules & skills:** Declarative rules live in `.agents/rules/` (auto-loaded by supporting harnesses or opened via the table links above); multi-step playbooks follow the Agent Skills layout in `.agents/skills/`.
