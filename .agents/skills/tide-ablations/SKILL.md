---
name: tide-ablations
description: Runs tide overfit ablations — confidence threshold sweep, loss-weight variants, and architecture changes on MERT. Use when investigating F1 plateau, loss weight tuning, threshold post-hoc, or arch_medium vs arch_large on tide.
disable-model-invocation: true
---

# Tide ablations

## Script

`scripts/run_overfit_tide_ablations.py`

Use the [WSL command template](../wsl-gpu-stepcovnet/SKILL.md#command-template) with:

`scripts/run_overfit_tide_ablations.py --epochs=50 --phase=all`

Phases:

| Phase       | What it does                                              |
| ----------- | --------------------------------------------------------- |
| `threshold` | Sweep confidence on existing MERT checkpoint (no retrain) |
| `train`     | Retrain MERT with loss-weight and architecture variants   |
| `all`       | Threshold sweep then training ablations (default)         |

## Output

| Artifact       | Path                                                      |
| -------------- | --------------------------------------------------------- |
| Summary        | `models_wsl/overfit_tide_ablations/ablation_summary.json` |
| Per-run models | `models_wsl/overfit_tide_ablations/<variant>/`            |
| Callbacks      | `callbacks/overfit_tide_ablations/<variant>/`             |

## EXP-10 conclusions (do not re-run blindly)

- **Threshold sweep:** no meaningful F1 lift — recall capped before post-hoc threshold helps
- **Loss weights** (cls-heavy, time-heavy, both-high): ~same ~28–30% F1
- **arch_medium** (embed 128): similar to baseline
- **arch_large** (embed 256): **OOM** on 8 GB GPU — skip or reduce batch

Plateau is **not** fixed by epochs, loss weights, threshold, or medium arch bump. Likely **query-slot formulation** (high FP, ~233/634 recall).

## After the run

1. Read `ablation_summary.json` (if script crashed, check terminal + partial models)
2. Log `EXP-…` in [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md); tag pipeline stages: `train` for loss, `model` for arch, `post` for threshold
3. Next: multi-song val or formulation alternatives per [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) § Recommended next step

## Known pitfall

`sweep_confidence_thresholds` requires numeric `min_onset_distance_ms` (use `0.0`, not `None`). See JRN-20260606-05 in [self-journal.md](../../../docs/agents/self-journal.md).
