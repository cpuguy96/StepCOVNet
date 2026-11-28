---
description: Design doc schema fields and declarative research logging constraints under docs/
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

## Research logging constraints (`docs/research/**`)

- Persist measurable runs (`EXP-YYYYMMDD-NN`), design tradeoffs/insights (`NOTE-YYYYMMDD-NN`), and prior-art citations (`PAPER_LEDGER.md` + `paper.bib`) in the **same turn** without asking for approval.
- Use `YYYY-MM-DD HH:MM:SS` from the system clock on every entry; cite prior art (never describe baselines as stealing).
- Grep `EXPERIMENT_LOG.md` and `logs/` before proposing a re-run; promote to `PAPER_OUTLINE.md` only when drafting.
- Call `reproducibility.apply_training_seed(seed)` before model construction and dataset creation.

Full procedure: [research-session-workflow](../skills/research-session-workflow/SKILL.md).
