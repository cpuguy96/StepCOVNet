# Tide overfit — version history

Frozen experiment recipes. **Do not edit** after a version ships; add `vN+1.json` instead.

**Champion (current best):** [`../../tide_overfit.json`](../../tide_overfit.json) — graduated from **v1** with canonical artifact paths and `val_overfit_gate` checkpointing.

## Promotion rule

1. Train/eval a candidate (`vN.json` or a local fork).
2. Offline gate must beat the champion on the primary metric: **free-run ordered 634/634 @ 20 ms** (`debug_ar_onset_overfit.py --ar_decode`). Teacher-only gains do not graduate.
3. Copy the winning `run` + `model` blocks into `configs/ar/tide_overfit.json`; set `model_output_dir` / `callback_root_dir` to `models_wsl/ar/tide_overfit` and `callbacks/ar/tide_overfit`.
4. Log EXP + update the table below.

## Versions

| Ver    | File               | Was                    | Teacher ordered | Free-run ordered   | Notes                                          |
| ------ | ------------------ | ---------------------- | --------------- | ------------------ | ---------------------------------------------- |
| **v1** | [v1.json](v1.json) | `tide.json` / gate_v5  | **634/634**     | 12/634 (early EOS) | **Champion base** — from scratch; teacher PASS |
| v2     | [v2.json](v2.json) | `overfit_perfect/base` | ~633/634        | ~619/634           | Warm-start v1; SS=0 polish phase A             |
| **v3** | [v3.json](v3.json) | `overfit_perfect/run2` | 632/634         | **612/634**        | **Best free-run** so far; λ_res=10, early stop |
| v4     | [v4.json](v4.json) | `overfit_perfect/run3` | —               | —                  | 200 ep polish from v3                          |
| v5     | [v5.json](v5.json) | `overfit_perfect/run4` | —               | —                  | + `lambda_incremental_consistency`             |
| v6     | [v6.json](v6.json) | `run4_smoke`           | —               | —                  | 20 ep λ_inc smoke                              |
| v7     | [v7.json](v7.json) | `overfit_perfect/run5` | 633/634         | 619/634            | v4 knobs, warm-start v3                        |

Artifact paths inside each `vN.json` are **historical** (reproduce old EXP logs). Only the champion uses `models_wsl/ar/tide_overfit/`.
