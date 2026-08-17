---
name: steering-correction-promotion
description: On user steering corrections or remember-this, pick the smallest durable artifact (skill section, scoped rule, or rare always-apply rule), slim AGENTS.md, optimize agent-brain context, then prepend JRN. Use instead of creating new alwaysApply rules by default.
disable-model-invocation: true
---

# Steering correction promotion

Run this skill **in the same turn** as any user steering correction — how you decide, operate, route, log, or commit. Finish with artifact + JRN + brief confirmation.

Also invoked from [agent-self-improvement](../agent-self-improvement/SKILL.md).

## Context budget (optimize every promotion)

| Layer                     | Cost                                                         | Use for                                                            |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `alwaysApply: true` rules | **Every turn** — add sparingly; merge or scope when possible | Routing, portable paths, Python env — only what _every_ task needs |
| `globs` rules             | **When matching files** are in play                          | Script execution, docs schema, test patterns                       |
| Skills                    | **When task matches** or agent opens skill                   | How to decide or run — not live F1 / EXP ids / today's Next action |
| Current phase             | Live scoreboard                                              | Next action, blockers, run numbers — [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) |
| `AGENTS.md`               | **Router only** — stay thin                                  | One index table; no procedure prose                                |
| Self-journal              | **Receipt** — not loaded every turn                          | Audit trail with **Artifact** path                                 |

**Default:** prefer **skill section** or **scoped rule** over a new `alwaysApply` rule. Put live experiment state in Current phase, not in the skill body.

## Decision tree

Answer in order; stop at the first fit.

```
1. Multi-step procedure (train, debug playbook, log EXP)?
   → Extend existing skill OR new skill + skills README row
   → AGENTS.md: new index row ONLY if nothing routes there yet

2. Only matters when editing/running certain paths?
   → Scoped rule (.mdc with globs, alwaysApply: false)
   → Merge into existing scoped rule if same glob family (e.g. scripts/**)

3. Universal but <5 lines and same topic as an existing alwaysApply rule?
   → Merge into that rule — do NOT add a new file

4. Truly required on EVERY turn (safety, entry routing)?
   → alwaysApply rule — LAST RESORT
   → Keep under ~40 lines; regenerate [agent-brain.md](../../../docs/agents/agent-brain.md)

5. One-off preference for this session only?
   → Chat — no JRN unless user says remember this

6. Mechanical bug with a code fix?
   → Code/tests first; optional scoped rule if agents keep repeating it
```

## Same-turn workflow

| Step | Action                                                                               |
| ---- | ------------------------------------------------------------------------------------ |
| 1    | Run **decision tree** — pick smallest enforcement layer                              |
| 2    | **Implement** — edit skill/rule/AGENTS/code; **dedupe** while editing                |
| 3    | **Optimize** — see checklist below                                                   |
| 4    | **JRN** — prepend with **Artifact** path(s)                                          |
| 5    | **Confirm** — artifact + journal id + what was _not_ promoted to alwaysApply         |
| 6    | **Quick refresh** — [agent-brain-refresh](../agent-brain-refresh/SKILL.md) steps 1–3 |

## Optimization checklist (required)

After every promotion, scan agent brain for waste:

- [ ] **No new alwaysApply** unless step 4 of decision tree — justify in JRN; prefer scoped rule or skill
- [ ] **Merge** rules that share a glob or topic (e.g. one `scripts/**` rule for shell output)
- [ ] **Demote** any alwaysApply rule that only applies to scripts, tests, or docs → set `globs`, `alwaysApply: false`
- [ ] **AGENTS.md** — index rows only; link [agent-brain.md](../../../docs/agents/agent-brain.md) for rule catalog; no always-on rule table
- [ ] **Skills README** — one row per skill; no duplicate of rule body
- [ ] **Do not** duplicate the same constraint in rule + skill + AGENTS.md — **one** enforcement home

## AGENTS.md rules

| Do                                                                            | Don't                                                         |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Add a row to the task index when a new skill/index is the entry               | Paste skill or rule content into AGENTS.md                    |
| Link [agent-brain.md](../../../docs/agents/agent-brain.md) for rule inventory | Duplicate alwaysApply lists (Cursor loads them automatically) |
| Link north star / pre-submit once                                             | Grow AGENTS.md with journal or EXP content                    |

## Rule catalog

Update [docs/agents/agent-brain.md](../../../docs/agents/agent-brain.md) by hand during refresh. Verify with read-only `python scripts/audit_agent_brain.py`. Scoped summary: [skills README § Agent brain](../README.md#agent-brain-scoped-rules).

## JRN receipt

```markdown
| **Artifact** | Path(s) — include layer chosen (skill / scoped rule / alwaysApply / code) |
| **Action taken** | One line: what enforces the correction + any demotion/merge done |
```

## Signals

Steering: “remember this”, “don’t …”, “always …”, “stop doing X”, corrections to agent _process_ (not hyperparameters or one-off task args).
