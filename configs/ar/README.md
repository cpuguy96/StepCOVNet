# AR onset configs

## Champion — tide overfit

| | |
| --- | --- |
| **Config** | [`tide_overfit.json`](tide_overfit.json) |
| **Train** | `python scripts/train_onset_ar.py --config configs/ar/tide_overfit.json` |
| **Verify** | `python scripts/debug_ar_onset_overfit.py --config configs/ar/tide_overfit.json --ar_decode` |
| **Artifacts** | `models_wsl/ar/tide_overfit/`, `callbacks/ar/tide_overfit/` |

Pass bar: **free-run ordered 634/634 @ 20 ms** ([ONSET_METRICS.md](../../docs/research/ONSET_METRICS.md)). Training checkpoints on `val_overfit_gate`; free-run is **offline only** (`ar_decode_val_every_n_epochs: 0`).

Graduated from **v1** (teacher-pass from scratch). Best experimental free-run remains **v3** (612/634) — see [versions/tide_overfit/README.md](versions/tide_overfit/README.md).

## Versioned experiments (do not delete)

| Directory | Contents |
| --------- | -------- |
| [`versions/tide_overfit/`](versions/tide_overfit/) | v1–v7 frozen tide overfit recipes + promotion table |
| [`versions/tide_overfit_decode/`](versions/tide_overfit_decode/) | Scheduled-sampling / decode warm-start experiments |

To try an old recipe: `--config configs/ar/versions/tide_overfit/v3.json` (uses historical `model_output_dir` in that file).

## Graduating a new champion

See [versions/tide_overfit/README.md](versions/tide_overfit/README.md#promotion-rule). Summary: beat current best on **offline free-run** ordered match → copy into `tide_overfit.json` → log EXP.
