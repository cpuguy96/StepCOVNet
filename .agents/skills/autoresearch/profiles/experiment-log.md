# Profile: experiment-log

General StepCOVNet research. Use for pipeline stages, ablations, or one-off scripts where success is defined in the user goal.

## Success criteria

**From the user goal.** Examples:

- Metric improves by X% on a named benchmark
- `EXP-…` hypothesis confirmed with numbers in [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md)
- Test suite passes with lower runtime

If the user omits a metric, ask **once** before the loop starts.

## Preflight

- One heavy GPU job at a time ([wsl-gpu-stepcovnet](../../wsl-gpu-stepcovnet/SKILL.md)).
- Read [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md) § Current phase.

## Orient

1. [docs/research/README.md](../../../../docs/research/README.md) — one linked doc
2. [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md) — recent `EXP-…` / phase
3. [DISCUSSION_NOTES.md](../../../../docs/research/DISCUSSION_NOTES.md) — open questions
4. Profile-specific artifacts (e.g. `ablation_summary.json`, prior checkpoints)

## Plan artifact

- **Tier A/B:** config path, CLI args, or notebook parameter block documented in chat + upcoming `EXP-…` draft
- **Tier C:** code edit plan + test list

## Run

User goal or standard script from [project layout](../../../../docs/agents/project-layout.md). Common entries:

| Task            | Entry                                           |
| --------------- | ----------------------------------------------- |
| ITGPT placement | [itgpt-onset](itgpt-onset.md) / `scripts/train_itgpt_placement.py` |
| Dense ablations | [tide-ablations](../../tide-ablations/SKILL.md) |

## Log

Follow [research-session-workflow](../../research-session-workflow/SKILL.md): prepend `EXP-…` and optional `NOTE-…`.

## Anti-spam

Same global rules: plateau → tier upgrade; no identical reruns; Tier C needs tests before long GPU.

## Gotchas

| Gotcha                             | What to do                                                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Re-running without checking log    | `grep` [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md) for checkpoint/config before proposing same job |
| Logging every run in PAPER_OUTLINE | Use EXPERIMENT_LOG; promote to paper only when drafting                                                                 |
