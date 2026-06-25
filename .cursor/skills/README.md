# Project skills

Procedure index for Cursor agents. Routed from [AGENTS.md](../../AGENTS.md).

**Skills = multi-step playbooks.** Short policies live in [`.cursor/rules/`](../rules/). Agent state (layout, journal, ownership): [docs/agents/README.md](../../docs/agents/README.md).

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
| F1 zero or low, conv1d collapse, train vs eval matching, threshold debug | onset-event-eval-matching | [onset-event-eval-matching/SKILL.md](onset-event-eval-matching/SKILL.md) |
| loss weights, arch ablation, threshold sweep, plateau investigation      | tide-ablations            | [tide-ablations/SKILL.md](tide-ablations/SKILL.md)                       |
| start research session, log EXP/NOTE                                     | research-session-workflow | [research-session-workflow/SKILL.md](research-session-workflow/SKILL.md) |
| WSL GPU training, MERT extract, TensorFlow CUDA on Windows               | wsl-gpu-stepcovnet        | [wsl-gpu-stepcovnet/SKILL.md](wsl-gpu-stepcovnet/SKILL.md)               |
| agent mistake, process improvement, new repeated workflow                | agent-self-improvement    | [agent-self-improvement/SKILL.md](agent-self-improvement/SKILL.md)       |

## Scripts without skills (yet)

| User / task trigger                                     | Entry                                                                                    |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| raw simfile → `final_data`, preprocess, dataset prep    | `scripts/preprocess_dataset.py`, [DATASET_PREP_PIPELINE.md](../../docs/research/DATASET_PREP_PIPELINE.md) |
| train/val split manifest for `final_data`             | `scripts/build_training_index.py`, [DATASET_PREP_PIPELINE.md](../../docs/research/DATASET_PREP_PIPELINE.md) §2 |
| dense / event train on `final_data`                     | `scripts/train_onset.py` or `train_onset_event.py` + `--training_index_path`, [wsl-gpu-stepcovnet](wsl-gpu-stepcovnet/SKILL.md) |
| bisection, half-cheat, grid oracle, smoke gate bug hunt | `scripts/run_overfit_tide_bisection.py`, [EXP-11](../../docs/research/EXPERIMENT_LOG.md) |

## Adding a skill

1. Create `.cursor/skills/<name>/SKILL.md` (Cursor create-skill format).
2. Add a row to **Playbooks** above.
3. Prepend [self-journal.md](../../docs/agents/self-journal.md) if the skill closes a documented gap.
