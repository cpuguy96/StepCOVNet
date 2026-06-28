---
name: research-session-workflow
description: Starts a research session and logs EXP and NOTE entries with timestamps. Use when beginning research work, logging experiments, or writing discussion notes. Promote findings to PAPER_OUTLINE only when drafting the paper.
disable-model-invocation: true
---

# Research session workflow

## Session start

1. [docs/research/README.md](../../../docs/research/README.md) — load **one** linked doc, not the full log
2. [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) § **Current phase** / **Recommended next step**
3. One `EXP-…` row from the same log index if needed
4. Skim [DISCUSSION_NOTES.md](../../../docs/research/DISCUSSION_NOTES.md) for open questions (newest first)

North star: [PIPELINE_ARCHITECTURE.md](../../../docs/research/PIPELINE_ARCHITECTURE.md).

## During work

Tag changes by pipeline stage: `pre` | `model` | `post` | `metric` | `train`.

Plan docs with JSON/schemas: [design-doc-fields.mdc](../../rules/design-doc-fields.mdc) and [DATASET_PREP_PIPELINE.md §6.3](../../../docs/research/DATASET_PREP_PIPELINE.md#63-schema-versions-and-field-rationale).

## Log experiment (`EXP-YYYYMMDD-NN`)

**Same turn, no ask:** After any measurable run or offline eval (`debug_ar_onset_overfit.py`, `eval_dense_onset.py`, threshold sweep, training finish), prepend or update [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) in the **same session** — do not offer to log or wait for user confirmation.

**Complete the series:** When a thread has multiple runs (run1, run2, v3, v4, offline `--ar_decode` logs under `logs/`), append **every** run’s key numbers to the same `EXP-…` entry in the session you discover them — not only the latest run and not vague “local v4” hand-waves.

**Before suggesting a re-run:** `grep` [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) and `logs/` for the checkpoint/config — if decode already exists on disk, read it and update the log instead of proposing the job again.

Prepend to [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) after measurable runs:

- Top of `## Experiment entries` + new index row at top
- **Timestamp:** `YYYY-MM-DD HH:MM:SS` from system clock (`Get-Date -Format "yyyy-MM-dd HH:mm:ss"` / `date +"%Y-%m-%d %H:%M:%S"`). No approximate suffixes.
- Track, status, config paths, key numbers, conclusion
- Increment `NN` per calendar day; check latest ID before prepending

Template: [_EXPERIMENT_ENTRY_TEMPLATE.md](../../../docs/research/_EXPERIMENT_ENTRY_TEMPLATE.md)

## Log discussion (`NOTE-YYYYMMDD-NN`)

Prepend to [DISCUSSION_NOTES.md](../../../docs/research/DISCUSSION_NOTES.md) for design insights, Q&A, failure analysis — even without a new run.

Fields: Timestamp, Topic, Context, Discovery, Implication, Open (optional), Related.

Template: [_DISCUSSION_NOTE_TEMPLATE.md](../../../docs/research/_DISCUSSION_NOTE_TEMPLATE.md)

## Update paper (optional — when drafting for publication)

Do **not** log every run in [PAPER_OUTLINE.md](../../../docs/research/PAPER_OUTLINE.md). [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) is the authoritative experiment record.

When a result is paper-worthy, promote it to the outline:

| Section | Action |
| ------- | ------ |
| Abstract / Results | Selected numbers and claims only |
| Methods | Align with PIPELINE_ARCHITECTURE.md |
| Discussion | Hypotheses supported by promoted EXP + NOTE |

Cite `EXP-…` / `NOTE-…`. Link back to EXPERIMENT_LOG for full history.

## Session end

- Prepend [self-journal.md](../../../docs/agents/self-journal.md) if a process gap was found → [agent-self-improvement](../agent-self-improvement/SKILL.md)
- After a major phase shift, refresh **Current phase** / **Recommended next step** at the top of [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) (same session as the `EXP-…` that changed routing)
