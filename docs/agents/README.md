# Agent state

Project context for Cursor agents — **not** research findings. Routed from [AGENTS.md](../../AGENTS.md).

## Files here

| File | When to read |
| ---- | ------------ |
| [project-layout.md](project-layout.md) | Locate code, configs, scripts, data, or model artifacts |
| [agent-brain.md](agent-brain.md) | Rules/skills catalog (agent-maintained; verify with `scripts/audit_agent_brain.py`) |
| [self-journal.md](self-journal.md) | Process mistakes, fixes, conventions, skill gaps |

## What lives elsewhere

| Question | Authoritative doc |
| -------- | ----------------- |
| What did we run / measure? | [EXPERIMENT_LOG.md](../research/EXPERIMENT_LOG.md) |
| What should we do next? | [whats-next skill](../../.cursor/skills/whats-next/SKILL.md) → [EXPERIMENT_LOG.md](../research/EXPERIMENT_LOG.md) § Current phase; local checks via `scripts/project_status.py` |
| Why did we decide X? | [DISCUSSION_NOTES.md](../research/DISCUSSION_NOTES.md) |
| Dataset prep / `final_data` | [DATASET_PREP_PIPELINE.md](../research/DATASET_PREP_PIPELINE.md) |
| Paper draft (optional) | [PAPER_OUTLINE.md](../research/PAPER_OUTLINE.md) — promote from log when drafting |
| Paper citations / claims | [PAPER_LEDGER.md](../research/PAPER_LEDGER.md) · [paper.bib](../research/paper.bib) |
| Target pipeline design | [PIPELINE_ARCHITECTURE.md](../research/PIPELINE_ARCHITECTURE.md) |
| AR onset (locked design, not implemented) | [AR_ONSET_DESIGN.md](../research/AR_ONSET_DESIGN.md) |
| Open decisions / gates | [DECISIONS_CHECKLIST.md](../research/DECISIONS_CHECKLIST.md) |
| How to run a procedure (train, overfit, WSL, debug) | [.cursor/skills/README.md](../../.cursor/skills/README.md) |
| Steering correction / agent brain optimization | [steering-correction-promotion/SKILL.md](../../.cursor/skills/steering-correction-promotion/SKILL.md) |
| Refresh agent brain / refresh all rules, audit rules and skills | [agent-brain-refresh/SKILL.md](../../.cursor/skills/agent-brain-refresh/SKILL.md) |
| Code style, tests, Python env | [agent-brain.md](agent-brain.md) — Cursor loads rules from `.cursor/rules/` |
