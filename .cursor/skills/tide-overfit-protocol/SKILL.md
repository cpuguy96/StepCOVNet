---
name: tide-overfit-protocol
description: Runs tide single-song overfit smoke tests across conv1d, mel, and MERT frontends via WSL GPU. Use when the user mentions tide overfit, overfit suite, frontend ablation, smoke test on tide, or comparing preprocessing frontends.
---

# Tide overfit protocol

## When to use

Single-song memorization check on **tide** before scaling to multi-song val. Default **50 epochs**; 100 ep only when explicitly testing epoch sensitivity.

## Preconditions

- Tide audio/chart paths in `configs/overfit_tide/*.json`
- MERT: `data/v2/test/tide.mert.npy` (or path in config `mert_features_dir`)
- Do **not** set `pipeline_check_shortcuts=true` unless explicitly debugging pipeline wiring

## Run (WSL GPU)

Use the [WSL command template](../wsl-gpu-stepcovnet/SKILL.md#command-template) with:

`scripts/run_overfit_tide_suite.py --epochs=50`

Subset frontends: `--frontends=conv1d,mel`

Scripts auto-dispatch from Windows when WSL is available.

## Configs and artifacts

| Item      | Path                                          |
| --------- | --------------------------------------------- |
| Configs   | `configs/overfit_tide/{conv1d,mel,mert}.json` |
| Models    | `models_wsl/overfit_tide/<frontend>/`         |
| Summary   | `models_wsl/overfit_tide/suite_summary.json`  |
| Callbacks | `callbacks/overfit_tide/<frontend>/`          |

## After the run

1. Read `suite_summary.json` for per-frontend F1 / precision / recall
2. Log `EXP-YYYYMMDD-NN` in [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) with full timestamp (index + entry)
3. If F1=0 or conv1d collapse → follow [onset-event-eval-matching](../onset-event-eval-matching/SKILL.md) before retraining
4. If plateau persists → [tide-ablations](../tide-ablations/SKILL.md) or formulation change per [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) § Recommended next step

## Related

- [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md)
- [PIPELINE_ARCHITECTURE.md](../../../docs/research/PIPELINE_ARCHITECTURE.md) — tag stage `pre` for frontend comparison
