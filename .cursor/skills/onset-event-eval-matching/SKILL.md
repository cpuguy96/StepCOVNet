---
name: onset-event-eval-matching
description: Diagnoses onset event F1, train/eval matching mismatches, confidence collapse, and threshold effects. Use when F1 is zero or low, conv1d collapse is suspected, or the user asks about Hungarian vs ordered assignment, matching tolerance, or debug_onset_overfit output.
disable-model-invocation: true
---

# Onset event eval and matching

## Core rule

**Training and eval must use the same assignment philosophy.** Eval always uses Hungarian matching with tolerance; training uses `assign_onset_pairs_l1` (Hungarian L1) in `src/stepcovnet/onset_events/losses.py`.

Ordered slot→GT pairing on tide (634 GT, 1024 uniform slots) can yield **zero** in-tolerance pairs → loss drives all confidences toward 0 → **0% F1** despite reasonable predicted times (see NOTE-20260606-13, EXP-07/08).

## Diagnostics-first (F1=0 or suspicious)

Before changing epochs, architecture, or loss weights, run diagnostics via the [WSL command template](../wsl-gpu-stepcovnet/SKILL.md#command-template) with:

`scripts/debug_onset_overfit.py --checkpoint=models_wsl/overfit_tide/<frontend>/onset_event_model.keras`

Check in output / `diagnostics.py`:

- Max confidence and count above 0.5
- Hungarian pairs within tolerance vs thresholded detections
- TP / FP / FN breakdown (expect ~233/634 TP at ~30% F1 plateau)

## Interpretation guide

| Symptom                         | Likely cause                                         | Next step                                                    |
| ------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Max conf ~0.0002, 0% F1         | Train/eval assignment mismatch or no learnable pairs | Verify Hungarian train loss; read NOTE-13                    |
| Many pairs in tolerance, low F1 | Threshold / post-hoc filtering                       | [tide-ablations](../tide-ablations/SKILL.md) threshold phase |
| ~28–30% F1, high FP             | Query-slot formulation cap, not metric bug           | Formulation change; see EXP-10                               |

## Key code paths

| Module                           | Role                                           |
| -------------------------------- | ---------------------------------------------- |
| `losses.py`                      | `assign_onset_pairs_l1` — training assignment  |
| `metrics.py`                     | `match_onsets_numpy` — eval Hungarian matching |
| `diagnostics.py`                 | Confidence sweeps, pair stats                  |
| `scripts/debug_onset_overfit.py` | Checkpoint inspection CLI                      |

## After diagnosis

Log insight as `NOTE-…` if design-relevant; `EXP-…` if a fix was validated by retrain.
