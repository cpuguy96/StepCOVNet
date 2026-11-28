---
name: agent-maintenance
description: Handles user steering corrections ("remember this", "always/don't ...") by promoting the smallest durable artifact with a JRN receipt, and audits/refreshes agent rules, skills, and AGENTS.md ("refresh agent brain", "refresh all rules", "audit rules").
---

# Agent maintenance

Two workflows for keeping agent behavior aligned with repo conventions and user steering.

## A. Steering correction promotion (same turn)

Run in the **same turn** whenever the user corrects agent *process* (how you decide, route, log, commit, or run tools) or says **remember this**.

### Context budget & decision tree

Pick the **smallest durable layer** (stop at the first match):

| Layer | Context cost | Use for |
| ----- | ------------ | ------- |
| **Skill section** (`.agents/skills/*/SKILL.md`) | Loaded on task match | Multi-step procedures, runbooks, tool commands, workflows |
| **Scoped rule** (`.agents/rules/*.md`, `trigger: glob`) | Loaded when matching files are touched | Path-scoped constraints (`**/*.py`, `docs/**`) |
| **Model-decision rule** (`.agents/rules/*.md`, `trigger: model_decision`) | Loaded when task description matches | Short declarative policies across non-file tasks (git/CI, Python env) |
| **Universal rule** (`trigger: always_on`) | Loaded every turn (**last resort**) | Only constraints truly unavoidable on every single turn |
| **Live scoreboard** ([EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) § Current phase) | On research orient | Live metrics, `EXP-…` IDs, today's next action (never put in rules/skills) |
| **Router** ([AGENTS.md](../../../AGENTS.md)) | Entry only | Task-index rows only — never paste rules or procedure prose here |
| **Code / tests** | Zero prompt cost | Mechanical bugs or checks enforceable in scripts/tests |

Keep **one** enforcement home (do not duplicate the same rule across `AGENTS.md`, rules, and skills). Put runbooks in skills, not rules.

### Same-turn steps

1. **Implement & dedupe** in the chosen artifact (plus tests/code if applicable).
2. **Prepend receipt** to [self-journal.md](../../../docs/agents/self-journal.md) with `Timestamp` (`YYYY-MM-DD HH:MM:SS` from system clock), `Artifact` path + layer, and one-line `Action taken`. Journal-only entries without an artifact are incomplete.
3. **Verify** with `python scripts/audit_agent_brain.py`.
4. **Confirm in chat**: artifact updated, `JRN-…` ID, and layer chosen.

## B. Brain refresh / audit

Run on **refresh agent brain**, **refresh all rules**, **audit rules**, or after structural changes to `.agents/`:

1. Refresh state (`git status`, `git ls-files`) and read each `.agents/rules/*.md` from disk to inspect scope, trigger fit (`model_decision` vs `glob` vs `always_on`), staleness, and overlap with skills or `AGENTS.md`.
2. Run `python scripts/audit_agent_brain.py` from repo root.
3. Fix any drift, broken links, dead-track references, bloated `always_on` rules, or runbooks sitting in rules instead of skills.
4. Re-run `python scripts/audit_agent_brain.py` until exit code `0`.
5. Prepend a `JRN-…` entry in [self-journal.md](../../../docs/agents/self-journal.md) only if material edits were made; otherwise report **clean — no edits**.
