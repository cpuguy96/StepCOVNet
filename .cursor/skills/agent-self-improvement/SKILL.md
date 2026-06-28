---
name: agent-self-improvement
description: On steering corrections and process learnings, create or update a rule, skill, or code artifact first, then prepend a self-journal receipt in the same turn. Journal alone is incomplete.
disable-model-invocation: true
---

# Agent self-improvement

## Same-turn promotion (required)

The journal is a **receipt**, not the enforcement layer. A process learning is **incomplete** until a durable artifact exists **in the same turn**.

| Step | Action |
| ---- | ------ |
| 1 | **Create or update the artifact** (rule, skill, code, or `AGENTS.md` row) |
| 2 | **Prepend JRN-…** citing the artifact path in **Artifact** |
| 3 | **Confirm in chat** — artifact path(s) + journal id |

**Order matters:** artifact **before** journal. Do not prepend a JRN with only a vague “Action taken” and plan to promote later.

### Pick an artifact

| Lesson type | Artifact |
| ----------- | -------- |
| Always-on constraint (every session) | `.cursor/rules/<name>.mdc` + row in `AGENTS.md` if new |
| Repeatable multi-step workflow | `.cursor/skills/<name>/SKILL.md` + skills README row |
| Mechanical / verifiable fix | Code or script + tests if applicable |
| One-off, no durable rule yet | **No journal** — chat is enough; or journal only after user asks to defer |

Journal-only entries are for **audit after promotion**, not a substitute for promotion.

## Two mechanisms

| Mechanism      | Location                                                            | Role                                      |
| -------------- | ------------------------------------------------------------------- | ----------------------------------------- |
| Rules / skills / code | `.cursor/rules/`, `.cursor/skills/`, repo code               | **Enforce** behavior in future sessions   |
| Self-journal   | [docs/agents/self-journal.md](../../../docs/agents/self-journal.md) | **Receipt** — what changed and why        |

**Rules** = always-on constraints. **Skills** = on-demand workflows. Do not duplicate rules into skills.

## “Remember this” (user key phrase)

1. **Artifact first** — rule, skill, or code per table above.
2. **Journal second** — prepend JRN with **Artifact** path.
3. **Confirm** artifact + journal id in chat.

Treat “remember this” as permission to update tracked agent docs without a separate commit request (user still controls git commits).

## Steering corrections (immediate — same turn)

When the user steers **how you decide or operate** — priorities, tooling, output visibility, when to ask vs act, commit policy, routing, etc.:

1. **Artifact first** (same turn).
2. **Journal second** — category `convention` or `mistake`.
3. **Confirm** artifact + journal id.

Signals: “remember this”, “don’t …”, “always …”, “I want … instead”, “stop doing X”, or any correction of agent *process* (not task parameters like hyperparameters).

## When to run the pipeline

**Immediately** (same turn):

- User **steering correction**
- **“Remember this”** and close variants
- A mistake that wasted time or misled conclusions
- A fix or workaround that should become default

**Periodically** during work (not only session end):

- After a discrete task where process mattered
- When the same friction appears twice in one session
- Before switching task areas if a lesson is not yet promoted

**Session end:** catch any trigger above that still has no artifact + JRN pair.

## JRN entry format

```markdown
### JRN-YYYYMMDD-NN: Short title

| Field            | Value                                            |
| ---------------- | ------------------------------------------------ |
| **Timestamp**    | YYYY-MM-DD HH:MM:SS (system clock at write time) |
| **Category**     | mistake \| fix \| convention \| skill-gap        |
| **Summary**      | What happened                                    |
| **Artifact**     | Path(s) created or updated (required)            |
| **Action taken** | One line: what the artifact enforces             |
| **Related**      | EXP-…, NOTE-…, skill name                        |
```

Insert at the **top** of `## Entries` in [self-journal.md](../../../docs/agents/self-journal.md). Increment `NN` per day.

## When to add or update a skill

Trigger: same workflow requested twice, or journal category `skill-gap`.

1. Create `.cursor/skills/<name>/SKILL.md`
2. Add row to [skills README](../README.md)
3. Optional routing hint in [AGENTS.md](../../../AGENTS.md)
4. Prepend JRN with **Artifact** = skill path

## Session-end checklist

- [ ] Research logged (`EXP-…` / `NOTE-…`) per [research-session-workflow](../research-session-workflow/SKILL.md)
- [ ] Every steering / process lesson this session → **artifact + JRN** (not JRN alone)
- [ ] New repeated workflow → skill + README + JRN
- [ ] Broken index links → fix immediately

## Do not

- Prepend a JRN without an **Artifact** path in the same turn (except explicit user deferral)
- Store research findings in the journal (use EXPERIMENT_LOG / DISCUSSION_NOTES)
- Create skills in `~/.cursor/skills-cursor/` (Cursor internal)
