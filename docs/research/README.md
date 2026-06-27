# Research documentation

Lab notebook and paper-oriented notes for StepCOVNet research (onset detection and related tracks).

**Agent entry:** [AGENTS.md](../../AGENTS.md) → this README for research tasks. Agent state (layout, journal): [docs/agents/README.md](../agents/README.md).

## Files

| File                                                                       | Role                                                          |
| -------------------------------------------------------------------------- | ------------------------------------------------------------- |
| [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)                       | **Target pipeline** — PRE → MODEL → POST → METRICS → feedback |
| [DATASET_PREP_PIPELINE.md](DATASET_PREP_PIPELINE.md)                       | **Raw simfile → `final_data`** — prep phases P0–P9 (complete) |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)                                     | **Authoritative** run log — § Current phase for routing (`EXP-YYYYMMDD-NN`) |
| [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md)                                 | Conversation insights — newest first (`NOTE-YYYYMMDD-NN`)     |
| [PAPER_OUTLINE.md](PAPER_OUTLINE.md)                                       | Paper draft skeleton — promote findings from log when drafting |
| [DECISIONS_CHECKLIST.md](DECISIONS_CHECKLIST.md)                           | Open decisions before ablation runs                           |
| [AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md)                                   | **Autoregressive onset** — v1 stack locked 2026-06; gates + slug registry (not implemented) |
| [../onset_events_plan.md](../onset_events_plan.md)                         | Historical event-onset WP plan (superseded for routing)       |

## For agents and contributors

Cursor rule [research-logging.mdc](../../.cursor/rules/research-logging.mdc) requires persisting **both** discussion and experiments here during research work, not only in chat.

**Workflow:**

1. **Discuss / explore** → prepend `NOTE-…` to `DISCUSSION_NOTES.md` (index row at top)
2. **Run / measure** → prepend `EXP-…` to `EXPERIMENT_LOG.md` (index row + entry)
3. **Design shifts** → update planning doc + [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) if stage contract changes
4. **Paper (optional)** → promote selected results to `PAPER_OUTLINE.md` only when drafting for publication

Logs are **newest-first** under each index; `Current phase` / `Recommended next step` in `EXPERIMENT_LOG.md` stay at the top for routing.
