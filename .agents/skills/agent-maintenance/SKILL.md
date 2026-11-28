---
name: agent-maintenance
description: Handles user steering corrections ("remember this", "always/don't ...") by promoting the smallest durable artifact with a JRN receipt, runs skill-local TDD hill-climbing evals (.agents/skills/<skill>/evals/suite.json), and audits/refreshes agent rules, skills, and AGENTS.md ("refresh agent brain", "refresh all rules", "audit rules").
---

# Agent maintenance

Three workflows for keeping agent behavior aligned with repo conventions, user steering, and local evals.

## A. Steering correction promotion (same turn)

Run in the **same turn** whenever the user corrects agent *process* (how you decide, route, log, commit, or run tools) or says **remember this**.

### Context budget & decision tree

Pick the **smallest durable layer** (stop at the first match):

| Layer | Context cost | Use for |
| ----- | ------------ | ------- |
| **Skill section** (`.agents/skills/<skill>/SKILL.md`) | Loaded on task match | Multi-step procedures, runbooks, tool commands, workflows |
| **Scoped rule** (`.agents/rules/*.md`, `trigger: glob`) | Loaded when matching files are touched | Path-scoped constraints (`**/*.py`, `docs/**`) |
| **Model-decision rule** (`.agents/rules/*.md`, `trigger: model_decision`) | Loaded when task description matches | Short declarative policies across non-file tasks (git/CI, Python env) |
| **Universal rule** (`trigger: always_on`) | Loaded every turn (**last resort**) | Only constraints truly unavoidable on every single turn |
| **Live scoreboard** ([EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) § Current phase) | On research orient | Live metrics, `EXP-…` IDs, today's next action (never put in rules/skills) |
| **Router** ([AGENTS.md](../../../AGENTS.md)) | Entry only | Task-index rows only — never paste rules or procedure prose here |
| **Code / tests** | Zero prompt cost | Mechanical bugs or checks enforceable in scripts/tests |

Keep **one** enforcement home (do not duplicate the same rule across `AGENTS.md`, rules, and skills). Put runbooks in skills, not rules.

### Same-turn steps

1. **Red/green eval (Workflow C below):** Add or tighten a failing eval case inside the target's local suite (`.agents/skills/<skill>/evals/suite.json` or `.agents/rules/evals/<rule>.json` — **local-only and gitignored; never commit or force-add prompt suites to git**).
2. **Implement & dedupe** in the chosen artifact (plus tests/code if applicable).
3. **Verify:** Run `python scripts/audit_agent_brain.py`, `python scripts/eval_agent_setup.py validate`, and run [run_workflow_eval.py](scripts/run_workflow_eval.py) against the baseline to confirm `FAIL -> PASS` with zero regressions.
4. **Prepend receipt** to [self-journal.md](../../../docs/agents/self-journal.md) with `Timestamp` (`YYYY-MM-DD HH:MM:SS` from system clock), `Artifact` path + layer, eval case ID(s), and one-line `Action taken`. Journal-only entries without an artifact are incomplete.
5. **Confirm in chat**: artifact updated, eval delta, `JRN-…` ID, and layer chosen.

## B. Brain refresh / audit

Run on **refresh agent brain**, **refresh all rules**, **audit rules**, or after structural changes to `.agents/`:

1. Refresh state (`git status`, `git ls-files`) and read each `.agents/rules/*.md` from disk to inspect scope, trigger fit (`model_decision` vs `glob` vs `always_on`), staleness, and overlap with skills or `AGENTS.md`.
2. Run `python scripts/audit_agent_brain.py` and `python scripts/eval_agent_setup.py validate` from repo root.
3. Fix any drift, broken links, dead-track references, bloated `always_on` rules, missing local eval suites, or runbooks sitting in rules instead of skills.
4. Re-run both checks until exit code `0`.
5. Prepend a `JRN-…` entry in [self-journal.md](../../../docs/agents/self-journal.md) only if material edits were made; otherwise report **clean — no edits**.

## C. Skill-local TDD hill-climbing

Whenever improving or adding a skill or rule (or fixing a trigger miss / behavioral slip):

1. **Add or tighten a local failing eval first:** Edit only the target skill's `.agents/skills/<skill>/evals/suite.json` (or `.agents/rules/evals/<rule>.json` — kept local and gitignored; never check eval prompts into git) to add/tighten `trigger_cases` and/or `behavior_cases`. Validate with `python scripts/eval_agent_setup.py validate --skill <skill>`.
2. **Establish baseline (must fail the new/tightened check):** Run [run_workflow_eval.py](scripts/run_workflow_eval.py) via `run_workflow` scoped to that skill (`Args={"skill": "<skill>", "tag": "base_<skill>"}`, or omit `"skill"` for a full-suite sweep). Outputs land in gitignored `logs/agent_evals/`.
3. **Edit the skill / rule** cleanly (e.g. sharpen frontmatter `description` for trigger F1, or clarify steps/constraints for behavior).
4. **Re-eval & hill-climb diff (must fix without regressions):** Re-run `run_workflow_eval.py` with `Args={"skill": "<skill>", "tag": "cand_<skill>", "baseline": "logs/agent_evals/base_<skill>.json"}` (or score via `python scripts/eval_agent_setup.py score --skill <skill> --predictions ... --baseline ...`). Require **`FIXED (FAIL -> PASS)`** on targeted case(s) and **`REGRESSED: none`** before committing.
