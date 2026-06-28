---
name: agent-brain-refresh
description: Audit and sync agent brain — read all rules from disk, reconcile skills/catalogs/routing. Triggers on refresh agent brain, refresh all rules, audit rules, or after steering promotion. Read-only audit script; agent writes doc updates.
disable-model-invocation: true
---

# Agent brain refresh

Holistic pass over **rules, skills, and routing docs** so they match disk and stay context-efficient.

## What “refresh all rules” means

User phrases **refresh all rules**, **refresh agent brain**, and **audit rules** all invoke this skill (full refresh).

| You do | You do not |
| ------ | ---------- |
| **Read every** `.cursor/rules/*.mdc` from disk this turn — re-ground on bodies, not only `agent-brain.md` or audit stdout | Assume summarized chat memory is current |
| Reconcile rules ↔ skills ↔ `AGENTS.md` ↔ catalogs; optimize scope and dedupe | Reload Cursor’s rule injection (alwaysApply is automatic) |
| Run `audit_agent_brain.py` for inventory + drift; **you** edit catalogs if needed | Let the script write markdown |
| Report **clean — no edits** when audit OK and content already agrees | Treat audit-only as sufficient without reading rule files |
| Edit rule/skill/router files only when judgment finds drift, waste, or duplication | Add always-on rules or load multiple `AGENTS.md` indexes |

**Rules are the center; the brain is the scope.** “All rules” means each `.mdc` is read and judged; refresh still covers skills and routing docs that must stay aligned with those rules.

## When to run

| Trigger | Action |
| ------- | ------ |
| **refresh all rules**, **refresh agent brain**, **audit rules**, **brain refresh** | Full refresh (below) |
| After [steering-correction-promotion](../steering-correction-promotion/SKILL.md) | Quick refresh (below) |
| Session end (with [agent-self-improvement](../agent-self-improvement/SKILL.md)) | Quick refresh if any brain file changed this session |
| Periodic | Every ~3 steering promotions or when `audit_agent_brain.py` reports drift |

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

### 1. Re-read all rules (agent)

Read **each** file in `.cursor/rules/*.mdc` (always-apply and scoped). Note scope, overlap with skills, and stale guidance.

### 2. Audit disk vs indexes (script)

```text
venv\Scripts\python.exe scripts/audit_agent_brain.py
```

Use stdout **disk inventory** as a checklist. Fix any **DRIFT** lines.

### 3. Update docs (agent — not the script)

| File                                                  | You edit when                         |
| ----------------------------------------------------- | ------------------------------------- |
| [agent-brain.md](../../../docs/agents/agent-brain.md) | Rules or skills added/removed/demoted |
| [skills README](../README.md)                         | Playbooks row + scoped-rules table    |
| [AGENTS.md](../../../AGENTS.md)                       | New routing index row only            |

### 4. Sync indexes

| Check                                     | Fix                                                         |
| ----------------------------------------- | ----------------------------------------------------------- |
| Each `*/SKILL.md` under `.cursor/skills/` | Row in skills README **Playbooks**                          |
| Each scoped `.mdc`                        | Row in skills README **Agent brain** table + agent-brain.md |
| New skill needs routing                   | One row in AGENTS.md task index only                        |
| Procedure text in AGENTS.md               | Move to skill; leave link                                   |

### 5. Optimize

- **Merge** scoped rules that share globs or topic
- **Demote** mistaken `alwaysApply: true` → set `globs`, `alwaysApply: false`
- **Avoid** adding always-on rules unless every task needs them
- **Delete** orphan rules; grep repo for stale filenames
- **Dedupe** — one enforcement home (rule, skill, or code — not all three)
- **Trim** AGENTS.md — router + pre-submit pointer only

### 6. Re-run audit

```text
venv\Scripts\python.exe scripts/audit_agent_brain.py
```

Exit 0 before finishing.

### 7. Report

- **Clean** or drift fixes (bullets)
- alwaysApply vs scoped counts (informational)
- Files you edited (or **none**)

### 8. Journal (if material changes)

Prepend JRN with **Artifact** paths when refresh changed rules, catalog, or indexes.

## Quick refresh (after promotion)

1. Skim changed `.mdc` / skill files
2. `audit_agent_brain.py` (read)
3. Update agent-brain.md / README if needed
4. Re-run audit

## Do not

- Use `audit_agent_brain.py` to write or overwrite markdown
- Add always-on rule lists back into AGENTS.md (link agent-brain.md instead)
- Create journal entries without artifact changes
