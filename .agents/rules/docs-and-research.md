---
description: Design doc schema fields and research logging rules under docs/
globs: docs/**
trigger: glob
---

# Docs and research logging

## Design and plan docs (`docs/**`)

When creating or updating design/plan docs under `docs/` that define dataclasses, JSON artifacts, manifests, or config surfaces:

1. **`schema_version`** on every JSON artifact + version changelog (what v1 is, what triggers v2+).
2. **Field rationale table** per object: `Field | Meaning | Decision (§/ID) | Why / notes`.
3. **Deferred fields** listed explicitly with a link to open items.
4. **Cross-links** between resolved decisions, field tables, and integration docs, plus CLI flag defaults, enum/status meanings, and validation gates.

Reference shape: [DATASET_PREP_PIPELINE.md §6.3](../../docs/research/DATASET_PREP_PIPELINE.md#63-schema-versions-and-field-rationale).

## Research logging (`docs/research/**`)

When work under `docs/research/` produces runs, design decisions, or durable insights, persist them in the **same turn** without asking the user to approve logging:

| Artifact | When | ID / key |
| -------- | ---- | -------- |
| `EXPERIMENT_LOG.md` | Measurable or reproducible run (training, eval, threshold sweep) | `EXP-YYYYMMDD-NN` |
| `DISCUSSION_NOTES.md` | Design Q&A, tradeoffs, failure analysis | `NOTE-YYYYMMDD-NN` |
| `PAPER_LEDGER.md` + `paper.bib` | Paper, codebase, pack, or metric used as prior art / baseline / method / data | citation key |
| `PAPER_OUTLINE.md` | Paper-worthy results only (optional; when drafting) | — |

- **Timestamps:** Use `YYYY-MM-DD HH:MM:SS` from the system clock at write time.
- **No duplicate runs:** Grep `EXPERIMENT_LOG.md` and `logs/` before proposing a re-run.
- **Prior art:** Cite prior work via `PAPER_LEDGER.md` + `paper.bib`; recreating a published pipeline is a baseline, not stealing.
- **Training seeds:** Call `reproducibility.apply_training_seed(seed)` before model construction and dataset creation.

Full workflow: [research-session-workflow](../skills/research-session-workflow/SKILL.md).
