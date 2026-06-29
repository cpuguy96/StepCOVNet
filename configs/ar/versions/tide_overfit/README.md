# Tide overfit — version history

Frozen experiment recipes. **Do not edit** after a version ships; add `vN+1.json` instead.

**Champion:** [`../../tide_overfit.json`](../../tide_overfit.json) · metrics [`../../tide_overfit.manifest.json`](../../tide_overfit.manifest.json)

## Promotion rule (mandatory)

Whenever a version **beats the champion** on the primary metric, graduate it **in the same session**:

1. Offline eval: `debug_ar_onset_overfit.py --config <candidate> --model_path <ckpt> --ar_decode`
2. Compare **`ar_decode.ordered_onset_match.rate`** to [`tide_overfit.manifest.json`](../../tide_overfit.manifest.json) (higher wins; tie-break `n_matched`, then `n_denom` closer to 634).
3. Run: `python scripts/graduate_ar_tide_overfit.py --config <candidate> --model-path <ckpt> --version-ref versions/tide_overfit/vN.json`
4. Log EXP + add a row below if the version file is new.

Teacher-only improvements **do not** count unless free-run rate is unchanged or better.

Ultimate **done** bar remains free-run **634/634 @ 20 ms**; the champion always tracks the **current leader**, even below 1.0.

## Versions

| Ver    | File               | Was                    | Teacher ordered | Free-run ordered   | Notes                                          |
| ------ | ------------------ | ---------------------- | --------------- | ------------------ | ---------------------------------------------- |
| v1     | [v1.json](v1.json) | `tide.json` / gate_v5  | **634/634**     | 12/634 (early EOS) | From scratch; teacher PASS                     |
| v2     | [v2.json](v2.json) | `overfit_perfect/base` | ~633/634        | ~619/634           | Warm-start v1; SS=0 polish phase A             |
| v3     | [v3.json](v3.json) | `overfit_perfect/run2` | 632/634         | 612/634            | λ_res=10, early stop                           |
| v4     | [v4.json](v4.json) | `overfit_perfect/run3` | —               | —                  | 200 ep polish from v3                          |
| v5     | [v5.json](v5.json) | `overfit_perfect/run4` | —               | —                  | + `lambda_incremental_consistency`             |
| v6     | [v6.json](v6.json) | `run4_smoke`           | —               | —                  | 20 ep λ_inc smoke                              |
| **v7** | [v7.json](v7.json) | `overfit_perfect/run5` | 633/634         | **619/634**        | **Current champion source** (graduated 2026-06-28) |

Artifact paths inside each `vN.json` are **historical**. The champion always writes to `models_wsl/ar/tide_overfit/`.
