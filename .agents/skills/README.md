# Project skills

Procedure index for AI agents. Routed from [AGENTS.md](../../AGENTS.md).

**Skills = multi-step playbooks.** Short policies live in [`.agents/rules/`](../rules/). Agent state (layout, journal, ownership): [docs/agents/README.md](../../docs/agents/README.md).

## Invocation

Most skills set `disable-model-invocation: true` — **open the matching `SKILL.md` explicitly** when the task fits. Only **tide-overfit-protocol** may auto-attach from description match.

| Skill                         | Auto-attach?                   |
| ----------------------------- | ------------------------------ |
| tide-overfit-protocol         | Yes                            |
| All others in the table below | No — load skill when triggered |

## Playbooks

| User / task trigger                                                      | Skill                     | Path                                                                     |
| ------------------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------------------ |
| tide overfit, overfit suite, frontend ablation, smoke test on tide       | tide-overfit-protocol     | [tide-overfit-protocol/SKILL.md](tide-overfit-protocol/SKILL.md)         |
| run autoresearch, research goal + time budget, one-prompt overnight loop | autoresearch              | [autoresearch/SKILL.md](autoresearch/SKILL.md)                             |
| ITGPT-MERT onset hill-climbing, parallel Colab/WSL experiments (profile) | autoresearch → itgpt-onset | [autoresearch/profiles/itgpt-onset.md](autoresearch/profiles/itgpt-onset.md) |
| F1 zero or low, conv1d collapse, train vs eval matching, threshold debug | onset-event-eval-matching | [onset-event-eval-matching/SKILL.md](onset-event-eval-matching/SKILL.md) |
| loss weights, arch ablation, threshold sweep, plateau investigation      | tide-ablations            | [tide-ablations/SKILL.md](tide-ablations/SKILL.md)                       |
| what to do now, what's next, where are we, project status, orientation; **do it** after whats-next | whats-next                  | [whats-next/SKILL.md](whats-next/SKILL.md)                               |
| start research session, log EXP/NOTE, add paper citations                  | research-session-workflow | [research-session-workflow/SKILL.md](research-session-workflow/SKILL.md) |
| WSL GPU training, MERT extract, PyTorch CUDA on Windows                  | wsl-gpu-stepcovnet        | [wsl-gpu-stepcovnet/SKILL.md](wsl-gpu-stepcovnet/SKILL.md)               |
| Colab GPU training, parallel cloud runs, private Drive sync (`rclone` / `colab` CLI) | colab-gpu-stepcovnet      | [colab-gpu-stepcovnet/SKILL.md](colab-gpu-stepcovnet/SKILL.md)           |
| agent mistake, process improvement, new repeated workflow                | agent-self-improvement    | [agent-self-improvement/SKILL.md](agent-self-improvement/SKILL.md)       |
| steering correction, remember this, optimize agent brain / context       | steering-correction-promotion | [steering-correction-promotion/SKILL.md](steering-correction-promotion/SKILL.md) |
| refresh all rules, refresh agent brain, audit rules/skills, sync catalogs | agent-brain-refresh       | [agent-brain-refresh/SKILL.md](agent-brain-refresh/SKILL.md)               |

## Agent brain (scoped rules)

Canonical catalog: [docs/agents/agent-brain.md](../../docs/agents/agent-brain.md) (agent-maintained; verify with `python scripts/audit_agent_brain.py`). Scoped rules load on file match — not every turn.

| Rule | Globs |
| ---- | ----- |
| [`scripts-execution.md`](../rules/scripts-execution.md) | `scripts/**` |
| [`design-doc-fields.md`](../rules/design-doc-fields.md) | `docs/**` |
| [`python-style.md`](../rules/python-style.md) | `**/*.py` |
| [`python-tests.md`](../rules/python-tests.md) | `{src,scripts,tests}/**/*` |
| [`research-logging.md`](../rules/research-logging.md) | `docs/research/**` |

Maintain this table during [agent-brain-refresh](agent-brain-refresh/SKILL.md).

## Scripts without skills (yet)

| User / task trigger                                     | Entry                                                                                    |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| raw simfile → `final_data`, preprocess, dataset prep    | `scripts/preprocess_dataset.py`, [DATASET_PREP_PIPELINE.md](../../docs/research/DATASET_PREP_PIPELINE.md) |
| train/val split manifest for `final_data`             | `scripts/build_training_index.py`, [DATASET_PREP_PIPELINE.md](../../docs/research/DATASET_PREP_PIPELINE.md) §2 |
| dense / arrow train on `final_data` | `scripts/train_onset.py` or `scripts/train_arrow.py`, [wsl-gpu-stepcovnet](wsl-gpu-stepcovnet/SKILL.md) |
| DDC / DDCL / ITGPT placement train & eval | `scripts/train_{ddc,ddcl,itgpt}_placement.py`, `scripts/eval_{ddc,ddcl,itgpt}_placement.py`, [wsl-gpu-stepcovnet](wsl-gpu-stepcovnet/SKILL.md) |
| what to do now / project orientation | `scripts/project_status.py`, [whats-next/SKILL.md](whats-next/SKILL.md) |

## Adding a skill

1. Create `.agents/skills/<name>/SKILL.md` (standard `SKILL.md` format with `name` and `description` frontmatter).
2. Add a row to **Playbooks** above.
3. Prepend [self-journal.md](../../docs/agents/self-journal.md) if the skill closes a documented gap.
