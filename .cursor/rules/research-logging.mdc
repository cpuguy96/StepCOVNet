---
description: Log EXP/NOTE and paper citations in docs/research/
globs: docs/research/**
alwaysApply: false
---

# Research logging

When work under `docs/research/` produces runs, design decisions, or durable insights — persist in the same session, not only in chat.

| Artifact              | When                                    | ID                 |
| --------------------- | --------------------------------------- | ------------------ |
| `EXPERIMENT_LOG.md`   | Measurable / reproducible run           | `EXP-YYYYMMDD-NN`  |
| `DISCUSSION_NOTES.md` | Design Q&A, tradeoffs, failure analysis | `NOTE-YYYYMMDD-NN` |
| `PAPER_LEDGER.md` + `paper.bib` | Paper, codebase, pack, or metric used as prior art / baseline / method / data | citation key |
| `PAPER_OUTLINE.md`    | Optional — promote paper-worthy results only | —            |

Prior art is **cited** (ledger key + BibTeX). Recreating a published pipeline is a baseline, not a substitute for a citation. Do not describe that as stealing.

**Timestamp** on every entry: `YYYY-MM-DD HH:MM:SS` from system clock at write time.

**Agent:** Log in the same turn — never ask the user to approve logging. Log all runs in a thread (including offline `--ar_decode` from `logs/`). Check the log and `logs/` before suggesting a re-run.

**Training seeds:** `reproducibility.apply_training_seed(seed)` before model construction and dataset creation.

Full workflow: [research-session-workflow skill](../../.cursor/skills/research-session-workflow/SKILL.md). Log runs in `EXPERIMENT_LOG.md`; citations in `PAPER_LEDGER.md` + `paper.bib`; `PAPER_OUTLINE.md` only when drafting the paper.
