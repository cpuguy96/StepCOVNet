# Agent self-journal

Iterative improvements to agent behavior, process, and conventions on this project. **Not** a substitute for [EXPERIMENT_LOG.md](../research/EXPERIMENT_LOG.md) or [DISCUSSION_NOTES.md](../research/DISCUSSION_NOTES.md) — those hold research findings; this holds **how we work better**.

**Order:** newest entries first (prepend below).

**Same-turn promotion:** The journal is a receipt, not the fix. Every entry must cite an **Artifact** path (rule, skill, or code) created or updated **in the same turn** before the JRN is written. See [agent-self-improvement skill](../../.cursor/skills/agent-self-improvement/SKILL.md).

## When to prepend

**Immediately** when the user makes a **steering correction** — how you decide, prioritize, run commands, log output, route docs, commit, etc. Do not wait for a long or “substantive” session to end.

**Also** when you discover:

- A mistake that wasted time or produced wrong conclusions
- A fix or workaround that should become default
- A user-established convention worth preserving
- A repeated workflow missing from `.cursor/skills/`

**Periodically** during work: after discrete tasks with process lessons, when the same friction recurs, or before switching task areas — journal in the same session; do not defer everything to session end.

Maintenance rule: **artifact first, journal second** in the same turn; link `EXP-…` / `NOTE-…` when relevant.

## Entry format

Insert **at the top** of [Entries](#entries) (below this section):

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

---

## Entries

### JRN-20260628-07: Audit script read-only; no alwaysApply budget

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: no script writes to agent brain docs; model updates catalogs; no hard alwaysApply cap but stay mindful of always-on rules.       |
| **Artifact**     | `scripts/audit_agent_brain.py`, `tests/audit_agent_brain_test.py`, `agent-brain-refresh/SKILL.md`, `steering-correction-promotion/SKILL.md` |
| **Action taken** | Script prints disk inventory + drift only; agent maintains agent-brain.md; alwaysApply budget removed from audit.                      |
| **Related**      | JRN-20260628-06, agent-brain-refresh                                                                                                  |

### JRN-20260628-06: Agent brain refresh skill and canonical catalog

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User wants holistic agent-brain maintenance; AGENTS.md wrongly listed scoped rules as always-on; periodic/user-triggered refresh.      |
| **Artifact**     | `.cursor/skills/agent-brain-refresh/SKILL.md`, `scripts/audit_agent_brain.py`, `docs/agents/agent-brain.md`; slimmed `AGENTS.md`      |
| **Action taken** | Catalog regenerated from disk; alwaysApply table removed from AGENTS.md; refresh after promotion + session end + user phrase.         |
| **Related**      | steering-correction-promotion, JRN-20260628-05                                                                                        |

### JRN-20260628-05: Context-efficient agent brain — promotion skill

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: alwaysApply rules for every steering correction wastes context; optimize agent brain on each correction via smallest durable layer. |
| **Artifact**     | `.cursor/skills/steering-correction-promotion/SKILL.md`, `.cursor/rules/scripts-execution.mdc` (scoped); demoted `long-running-console`, `temp-artifacts` |
| **Action taken** | Decision tree favors skills/scoped rules; AGENTS.md slimmed; promotion skill runs optimization checklist each correction.               |
| **Related**      | agent-self-improvement, JRN-20260628-04                                                                                               |

### JRN-20260628-04: Systematic temp file handling

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | Agents left `tmp_*` at repo root from command redirects; user asked for a systematic policy for any temp output.                      |
| **Artifact**     | `.cursor/rules/temp-artifacts.mdc` (later merged into `scripts-execution.mdc`), `.gitignore`, `AGENTS.md`, `docs/agents/project-layout.md` |
| **Action taken** | Captures go to `logs/` or `_tmp/` (gitignored); delete `_tmp` when done; never commit or use repo-root `tmp_*`.                       |
| **Related**      | long-running-console, JRN-20260628-01, temp-artifacts                                                                               |

### JRN-20260628-03: Same-turn promotion — artifact before journal

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | Journal entries are useless without durable promotion; user wants artifact (rule/skill/code) created first, JRN second, same turn.   |
| **Artifact**     | `.cursor/skills/agent-self-improvement/SKILL.md`, `docs/agents/self-journal.md`, `.cursor/rules/agents-entry.mdc`                     |
| **Action taken** | Enforces artifact-first workflow; JRN requires **Artifact** path; journal-only entries forbidden except explicit user deferral.      |
| **Related**      | JRN-20260628-02, agent-self-improvement                                                                                               |

### JRN-20260628-02: Journal on steering corrections, not only long sessions

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User wants self-journal updated periodically and **immediately** on steering corrections (how the agent decides/operates), not batched only after long sessions. |
| **Artifact**     | `.cursor/skills/agent-self-improvement/SKILL.md`, `docs/agents/self-journal.md`, `.cursor/rules/agents-entry.mdc`                     |
| **Action taken** | Journal on steering corrections in same turn; periodic journaling during work.                                                        |
| **Related**      | agent-self-improvement, JRN-20260628-01                                                                                               |

### JRN-20260628-01: Training output hidden in log files

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28 12:00:00                                                                                                                   |
| **Category**     | convention                                                                                                                            |
| **Summary**      | Agent ran WSL GPU training/decode with `*> logs/...` or `Tee-Object`, so the user could not watch epoch progress or errors live.      |
| **Artifact**     | `.cursor/rules/long-running-console.mdc`, `.cursor/skills/agent-self-improvement/SKILL.md`, `AGENTS.md`                               |
| **Action taken** | Long-running jobs must stream to visible terminal; optional log file via tee; remember-this trigger documented.                         |
| **Related**      | agent-self-improvement, wsl-gpu-stepcovnet                                                                                            |

### JRN-20260606-09: Dense seed after model init caused train lottery

| Field            | Value                                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Timestamp**    | 2026-06-06 16:54:25                                                                                                                                                |
| **Category**     | mistake                                                                                                                                                            |
| **Summary**      | `tf.random.set_seed` ran in `_fit_and_save_model` after `build_unet_wavenet_model`, so seed 42 in config did not control initialization — EXP-12 vs EXP-13/14 gap. |
| **Action taken** | Added `reproducibility.apply_training_seed()` before model build; repro gate script; EXP-15 logged.                                                                |
| **Related**      | EXP-15, NOTE-20260606-16, `check_dense_mert_reproducibility.py`                                                                                                    |

### JRN-20260606-08: Timestamps from system clock only

| Field            | Value                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 16:37:35                                                                                                                                   |
| **Category**     | convention                                                                                                                                            |
| **Summary**      | User required full `YYYY-MM-DD HH:MM:SS` on all log timestamps, captured from the machine clock at write time — no `(approx.)` or estimated suffixes. |
| **Action taken** | Updated `research-notebook.mdc`, skills, templates, and agent docs; stripped `(approx.)` from existing EXP/NOTE/JRN rows.                             |
| **Related**      | NOTE-20260606-15, `research-session-workflow` skill                                                                                                   |

### JRN-20260606-07: Newest-first research logs

| Field            | Value                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 18:00:00                                                                                                        |
| **Category**     | convention                                                                                                                 |
| **Summary**      | EXP/NOTE/JRN entries and index tables were append-only (oldest at top), forcing scroll to see latest runs.                 |
| **Action taken** | Reordered logs newest-first; updated `research-notebook.mdc`, templates, and skills to **prepend** entries and index rows. |
| **Related**      | `research-session-workflow` skill, NOTE-20260606-09                                                                        |

### JRN-20260606-06: Skills index without SKILL.md files

| Field            | Value                                                                                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 16:45:00                                                                                                                                               |
| **Category**     | skill-gap                                                                                                                                                         |
| **Summary**      | Initial agent-self-improvement pass created `skills-index.md` and routing updates but did not write `.cursor/skills/*/SKILL.md` bodies — index links were broken. |
| **Action taken** | Created all six project skills under `.cursor/skills/`; verify [skills README](../../.cursor/skills/README.md) after any index change.                             |
| **Related**      | `agent-self-improvement` skill, [skills README](../../.cursor/skills/README.md)                                                                                   |

### JRN-20260606-05: Ablation threshold sweep None min_gap TypeError

| Field            | Value                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 15:00:00                                                                                                                                              |
| **Category**     | fix                                                                                                                                                              |
| **Summary**      | `sweep_confidence_thresholds` passed `min_onset_distance_ms=None` into code expecting a float → `TypeError` during ablation threshold phase.                     |
| **Action taken** | Fixed in `diagnostics.py` (coerce None → 0.0); partial `ablation_summary.json` saved after arch_large OOM. Documented EXP-10 outcomes in `tide-ablations` skill. |
| **Related**      | EXP-10, `tide-ablations` skill                                                                                                                                   |

### JRN-20260606-04: F1=0 without running diagnostics first

| Field            | Value                                                                                                                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 14:30:00                                                                                                                                                                           |
| **Category**     | convention                                                                                                                                                                                    |
| **Summary**      | When conv1d showed 0% F1, the first instinct was to change epochs or architecture. Diagnostics (`debug_onset_overfit.py`, confidence stats) revealed assignment collapse, not a broken model. |
| **Action taken** | `onset-event-eval-matching` skill: **run diagnostics before retrain** when F1=0 or suspiciously low.                                                                                          |
| **Related**      | NOTE-20260606-13, `scripts/debug_onset_overfit.py`, `diagnostics.py`                                                                                                                          |

### JRN-20260606-03: WSL GPU env not sourced — silent CPU fallback

| Field            | Value                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 13:00:00                                                                                                                                        |
| **Category**     | mistake                                                                                                                                                    |
| **Summary**      | TensorFlow training launched in WSL without `source scripts/wsl_gpu_env.sh`. TF saw zero GPUs and trained on CPU with no obvious error — long runs wasted. |
| **Action taken** | Documented mandatory `wsl_gpu_env.sh` in `wsl-gpu.mdc` and `wsl-gpu-stepcovnet` skill; all WSL command templates include `source` before Python.           |
| **Related**      | `wsl-gpu-stepcovnet` skill, `wsl-gpu.mdc`                                                                                                                  |

### JRN-20260606-02: Ordered training vs Hungarian eval collapsed conv1d F1

| Field            | Value                                                                                                                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 14:12:41                                                                                                                                                                                                                                        |
| **Category**     | mistake                                                                                                                                                                                                                                                    |
| **Summary**      | Training used ordered slot→GT pairing while eval used Hungarian matching. On tide (634 GT, 1024 uniform slots), zero ordered pairs fell within 20 ms tolerance → loss pushed all confidences toward 0 → **0% F1** on conv1d despite reasonable pred times. |
| **Action taken** | Switched training to `assign_onset_pairs_l1` (Hungarian L1) in `losses.py`; logged EXP-08; created `onset-event-eval-matching` skill with diagnostics-first playbook.                                                                                      |
| **Related**      | EXP-07, EXP-08, NOTE-20260606-13                                                                                                                                                                                                                           |

### JRN-20260606-01: Wrong calendar dates in research logs

| Field            | Value                                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 10:30:00                                                                                                                                     |
| **Category**     | mistake                                                                                                                                                 |
| **Summary**      | Early EXP/NOTE entries used placeholder or incorrect dates (e.g. 2025) instead of the session calendar day. Broke ID consistency and paper cross-links. |
| **Action taken** | Renumbered IDs to `EXP-20260606-*` / `NOTE-20260606-*`; required full `YYYY-MM-DD HH:MM:SS` timestamps in `research-notebook.mdc`.                      |
| **Related**      | `research-session-workflow` skill, `research-notebook.mdc`                                                                                              |
