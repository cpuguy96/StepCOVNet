---
name: whats-next
description: Answers what to do now, what's next, where are we, project status, or orientation. When the user then says do it, go ahead, or start, execute the If you say do it steps from the last whats-next answer. Runs project_status.py and reads EXPERIMENT_LOG § Current phase.
---

# What's next

Use when the user asks **what to do now**, **what's next**, **where are we**, **status**, or **orientation** — including repeated asks across sessions.

Also load this skill when the user says **do it**, **go ahead**, **start it**, or **run it** after a whats-next answer (see [§ "Do it"](#do-it) below).

## Workflow (same turn)

1. Run status (repo root, Windows venv):

   ```text
   venv\Scripts\python.exe scripts/project_status.py
   ```

   Use `--json` only when merging into another tool.

2. Read [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) § **Current phase** — compact block (**Next action**, **Blockers**, **Defer until**) plus detail tables if needed.

3. If `project_status.py` lists **Missing prerequisites**, fold them into **Before you start** — do not suggest launching a run until blockers are addressed or the user accepts the risk.

4. If **GPU lock: held**, say what job holds it; do not start another WSL GPU train/MERT job.

5. Skim the latest `EXP-…` row in the experiment index only when status or Current phase references an incomplete run.

6. Reply using the **Answer template** below. One primary action — not a menu unless the user asked for options.

7. **Always** fill **If you say "do it"** with numbered, copy-paste-ready steps (commands, config paths, log names). This block is the contract for a follow-up **do it**.

## Answer template

```markdown
## Now
<One sentence — single best next action.>

## Why
- <2–3 bullets from Current phase / latest EXP — why this is the gate.>

## Before you start
- <Local blockers: missing manifests, GPU busy, dirty tree worth noting, memory mitigation. Use "None" when clear.>

## If you say "do it"
I will:
1. <First concrete step — exact command or file edit.>
2. <Next step — prerequisites before the main job.>
3. <Main job — train/extract/eval with config path and tee log path.>
4. <Post-run — offline decode, EXP log, etc., when part of the gate.>

**Done when:** <One measurable exit criterion — e.g. R2 completes 500 ep or ES; val teacher F1 logged.>

## After that
- <1–2 sequenced follow-ups from Current phase / ladder.>

## Alternate
- <One line — alternate track (e.g. dense scoreboard) if the user might switch.>
```

Keep prose tight. Link configs and docs with repo-relative paths.

## "Do it"

**Meaning:** Execute the numbered steps under **If you say "do it"** from the **most recent whats-next answer in this thread** — not the Alternate track, not a new guess.

| Situation | Action |
| --------- | ------ |
| User says **do it** / **go ahead** / **start it** / **run it** and the last assistant message had **If you say "do it"** | Execute those steps in order; do not re-ask for confirmation unless a step is destructive or ambiguous. |
| User says **do it** but there is no prior whats-next answer in the thread | Run [Workflow](#workflow-same-turn) first, present the full template (including **If you say "do it"**), then **immediately begin step 1** in the same turn unless a blocker requires user input (e.g. WSL memory change outside the repo). |
| User says **do the alternate** / **track A instead** | Execute the **Alternate** track, not **If you say "do it"**. |
| **Before you start** listed blockers | Resolve them as steps 1…N of **If you say "do it"** before the main job — already included in the numbered list. |

**Execution rules:**

- Follow [scripts-execution.mdc](../../rules/scripts-execution.mdc): visible terminal, tee to `logs/<topic>_<run>.log`, TensorBoard before/with GPU training.
- Follow [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md): one WSL GPU job at a time; check GPU lock first.
- Follow [research-session-workflow](../research-session-workflow/SKILL.md): log `EXP-…` when a run finishes or aborts; update Current phase compact block if routing changes.
- Reuse the same `model_output_dir` / `callback_root_dir` on reruns unless the user asked for a separate artifact tree.

**After execution:** Report what ran, where logs/checkpoints landed, and whether **Done when** is satisfied or what remains.

## Maintenance (agents)

When logging an `EXP-…` that changes routing, update the compact block at the top of § Current phase **in the same turn**:

| Field | When to change |
| ----- | -------------- |
| **Updated** | Any routing edit |
| **Next action** | New gate or rerun |
| **Blockers** | Abort reason, missing artifact, env ceiling |
| **Defer until** | Items explicitly waiting on the current gate |
| **Primary track** / **Alternate track** | Track A vs B swap |

See [research-session-workflow § Session end](../research-session-workflow/SKILL.md#session-end).

## Related

- Strategic record: [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md)
- AR ladder protocol: [AR_SCALING_LADDER.md](../../../docs/research/AR_SCALING_LADDER.md)
- Local checks: [scripts/project_status.py](../../../scripts/project_status.py)
