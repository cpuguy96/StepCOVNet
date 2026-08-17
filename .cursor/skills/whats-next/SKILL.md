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

## Evidence before recommending a run

**Do not** put a new train/probe/loss in **Now** (or **If you say "do it"**) unless you have **concrete evidence** it addresses the binding failure — not only that Current phase / a NOTE named it as a backlog idea.

| Allowed as **Now** | Not enough alone |
| ------------------ | ---------------- |
| Cheap diagnostics (ablation numbers, patch-acc curves, NLL vs uniform, error-mode analysis on an existing ckpt) | “Next action” text that lists an untested lever |
| A one-knob change with a mechanism already supported by logged numbers | Recipe spam (another λ / STE / aux) after similar recipes failed |
| A fix that already has unit/offline proof in-repo | Speculative architecture (“attn-mass might help”) with no supporting measurement |

If evidence is thin: **Now** = the measurement that would raise confidence; put speculative trains under **After that** or **Alternate**, and say confidence is low.

Copying Current phase **Next action** verbatim is wrong when that line is an unproven hypothesis — treat it as a candidate, then verify or demote it.

### No ladder scale-up unless asked

Do **not** put R3+ / more train songs / “scale the ladder” in **Now** or **If you say "do it"** unless the user **explicitly** asks to scale. Prefer fixed-R2 diagnostics, architecture, or objective changes. Scale may appear only under **Alternate** (and only if evidence supports it).

### No hard locality hacks as the product path

Hard prev-relative windows (`pointer_local_ce_radius` / force-advance / min_ahead) are **diagnostic crutches**: they shrink the search space using chart gap statistics and **fail on long pauses**. Do **not** put further R-shrink, force-advance, or similar hard decode/CE masks in **Now** / **do it** as the long-term approach — even if they raise val numbers.

| Allowed | Not the goal |
| ------- | ------------ |
| Use hard-R results as **evidence** that diffuse full-suffix CE was the failure mode | Treat hard-R / force-advance as the system to ship or keep stacking |
| Prefer objectives/architecture that **learn** to localize under mono/full support and still handle long gaps | Optimize R for this val set’s gap histogram |

If the user asks for another hard-mask probe anyway, run it under **Alternate** and label it diagnostic-only.

### Literature chain: DDC → DDCL → ITGPT

The research goal is **recreate published chart generation, then show an increment** on the matching corpus, split, grid, and metric ([NOTE-20260814-01](../../../docs/research/DISCUSSION_NOTES.md#note-20260814-01-literature-recreation-before-incremental-claims)). After Dataset A DDC placement T-repro is **accepted**:

| Now | Not Now |
| --- | ------- |
| Recreate **DDCL** on Dataset A (`omalley2025ddcl`): 48-slot placement has a real `M-slot48` number ([EXP-20260816-02](../../../docs/research/EXPERIMENT_LOG.md#exp-20260816-02-ddcl-48-slot-placement-full-split-on-dataset-a)); next is their audio-in-selection | `final_data` dense / MERT / AR as the literature scoreboard |
| Then **ITGPT** on Dataset B (`omalley2026itgpt`, expanded Fraxtil) | Mixing `M-ddc-20ms` with `M-slot48` in one table |
| Cite keys; match the paper’s PRE / metric | Treating ITL/Mizuki as “proper” until C5 |

Do **not** put DDC’s 256-class LSTM selection in **Now** just because C-LSTM placement was close enough. DDCL selection is in scope **as part of DDCL recreation**, after 48-slot placement exists — not as a DDC-peak choreography head.

### No `final_data` as literature comparison unless asked

Do **not** put subset or full-index `final_data` dense/MERT trains in **Now** / **do it**. That corpus is transfer/generalization only (C4). We have not shown it is a valid stand-in for Fraxtil/ITG.

### No more DDC placement eval unless asked

After Dataset A T-repro is **accepted**, or the user says DDC placement eval is **done**, do **not** put further DDC C-LSTM eval diagnostics in **Now** / **do it**.

| Allowed | Not until asked |
| ------- | --------------- |
| Citing logged F-score_c / F-score_m; a placement **train** the user requested | `timing_match`, matched-count, FP histograms, threshold sweeps, last-vs-best on the frozen ckpt |
| Starting **DDCL** 48-slot work | Stacking eval columns because ordered match is at the floor |

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
