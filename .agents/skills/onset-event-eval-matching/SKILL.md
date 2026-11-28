---
name: onset-event-eval-matching
description: Diagnoses onset event F1, train/eval matching mismatches, confidence collapse, and threshold effects. Use when F1 is zero or low, conv1d collapse is suspected, or the user asks about Hungarian vs ordered assignment, matching tolerance, or debug_onset_overfit output.
disable-model-invocation: true
---

# Onset event eval and matching

## Core rule

**Training and eval must use the same assignment philosophy.** Eval always uses Hungarian matching with tolerance (see `src/stepcovnet/onset_events/matching.py` and `metrics.py`).

Ordered slot→GT pairing can yield **zero** in-tolerance pairs → loss drives all confidences toward 0 → **0% F1** despite reasonable predicted times (see NOTE-20260606-13, EXP-07/08).

## Key code paths

| Module                              | Role                                           |
| ----------------------------------- | ---------------------------------------------- |
| `src/stepcovnet/onset_events/matching.py` | Core Hungarian / bipartite onset matching |
| `src/stepcovnet/onset_events/metrics.py`  | Event F1 and timing tolerance scoring     |

## After diagnosis

Log insight as `NOTE-…` if design-relevant; `EXP-…` if a fix was validated by retrain.
