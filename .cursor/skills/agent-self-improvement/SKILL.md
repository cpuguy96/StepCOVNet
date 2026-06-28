---
name: agent-self-improvement
description: Triggers on process mistakes and steering corrections; run steering-correction-promotion for artifact choice, context optimization, and JRN receipt. Journal alone is incomplete.
disable-model-invocation: true
---

# Agent self-improvement

## Steering corrections → promotion skill

On user steering, **remember this**, or a process mistake worth preserving:

1. Open and follow [steering-correction-promotion](../steering-correction-promotion/SKILL.md) **same turn**.
2. If brain files changed, run quick [agent-brain-refresh](../agent-brain-refresh/SKILL.md).

## Journal = receipt

| Step | Action |
| ---- | ------ |
| 1 | Artifact per promotion skill |
| 2 | Prepend JRN with **Artifact** path |
| 3 | Confirm artifact + journal id in chat |

Journal-only entries are incomplete (except explicit user deferral).

## When to run

**Immediately:** steering correction, remember this, costly mistake, durable fix.

**Periodically:** after discrete tasks with process lessons; before switching task areas.

**Session end:** catch missed artifact + JRN pairs; run [agent-brain-refresh](../agent-brain-refresh/SKILL.md) if brain files changed.

## JRN format

See [steering-correction-promotion § JRN receipt](../steering-correction-promotion/SKILL.md#jrn-receipt) and [self-journal.md](../../../docs/agents/self-journal.md).

## Do not

- Store research findings in the journal (use EXPERIMENT_LOG / DISCUSSION_NOTES)
- Create skills in `~/.cursor/skills-cursor/` (Cursor internal)
