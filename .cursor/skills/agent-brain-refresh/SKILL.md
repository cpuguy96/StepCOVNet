---
name: agent-brain-refresh
description: Audit and sync agent brain files — rules catalog, skills README, AGENTS.md router, stale links. Run when user asks to refresh agent brain, after steering promotion, or at session end. Uses read-only scripts/audit_agent_brain.py; agent writes doc updates.
disable-model-invocation: true
---

# Agent brain refresh

Holistic pass over **rules, skills, and routing docs** so they match disk and stay context-efficient.

## When to run

| Trigger                                                                          | Action                                                                    |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| User says **refresh agent brain**, **audit rules**, **brain refresh**            | Full refresh (this skill)                                                 |
| After [steering-correction-promotion](../steering-correction-promotion/SKILL.md) | Quick refresh (steps 1–4)                                                 |
| Session end (with [agent-self-improvement](../agent-self-improvement/SKILL.md))  | Quick refresh if any brain file changed this session                      |
| Periodic                                                                         | Every ~3 steering promotions or when `audit_agent_brain.py` reports drift |

## Architecture (do not duplicate)

| File                                                              | Role                                                                        |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `.cursor/rules/*.mdc`                                             | **Source of truth** for behavior — Cursor loads `alwaysApply` automatically |
| [docs/agents/agent-brain.md](../../../docs/agents/agent-brain.md) | **Human catalog** — **you** update during refresh; not script-generated     |
| [AGENTS.md](../../../AGENTS.md)                                   | **Router only** — task index; link to catalog                               |
| [.cursor/skills/README.md](../README.md)                          | Skill index + scoped-rules summary                                          |
| `scripts/audit_agent_brain.py`                                    | **Read-only** — prints disk inventory + drift; does not write files         |

**Minimize `alwaysApply` rules** — each one loads every turn. Prefer scoped rules (`globs`) or skills. No hard budget, but justify new always-on rules in JRN.

## Refresh workflow

### 1. Read disk (script)

```text
venv\Scripts\python.exe scripts/audit_agent_brain.py
```

Use stdout **disk inventory** as ground truth. Fix any **DRIFT** lines.

### 2. Update docs (agent — not the script)

| File                                                  | You edit when                         |
| ----------------------------------------------------- | ------------------------------------- |
| [agent-brain.md](../../../docs/agents/agent-brain.md) | Rules or skills added/removed/demoted |
| [skills README](../README.md)                         | Playbooks row + scoped-rules table    |
| [AGENTS.md](../../../AGENTS.md)                       | New routing index row only            |

### 3. Sync indexes

| Check                                     | Fix                                                         |
| ----------------------------------------- | ----------------------------------------------------------- |
| Each `*/SKILL.md` under `.cursor/skills/` | Row in skills README **Playbooks**                          |
| Each scoped `.mdc`                        | Row in skills README **Agent brain** table + agent-brain.md |
| New skill needs routing                   | One row in AGENTS.md task index only                        |
| Procedure text in AGENTS.md               | Move to skill; leave link                                   |

### 4. Optimize

- **Merge** scoped rules that share globs or topic
- **Demote** mistaken `alwaysApply: true` → set `globs`, `alwaysApply: false`
- **Avoid** adding always-on rules unless every task needs them
- **Delete** orphan rules; grep repo for stale filenames
- **Dedupe** — one enforcement home (rule, skill, or code — not all three)
- **Trim** AGENTS.md — router + pre-submit pointer only

### 5. Re-run audit

```text
venv\Scripts\python.exe scripts/audit_agent_brain.py
```

Exit 0 before finishing.

### 6. Report

- Drift fixes (bullets)
- alwaysApply vs scoped counts (informational)
- Files you edited

### 7. Journal (if material changes)

Prepend JRN with **Artifact** paths when refresh changed rules, catalog, or indexes.

## Quick refresh (after promotion)

1. `audit_agent_brain.py` (read)
2. Update agent-brain.md / README if needed
3. Re-run audit

## Do not

- Use `audit_agent_brain.py` to write or overwrite markdown
- Add always-on rule lists back into AGENTS.md (link agent-brain.md instead)
- Create journal entries without artifact changes
