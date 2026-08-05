# AR onset configs

## Champion — tide overfit

| | |
| --- | --- |
| **Config** | [`tide_overfit.json`](tide_overfit.json) |
| **Metrics** | [`tide_overfit.manifest.json`](tide_overfit.manifest.json) |
| **Train** | `python scripts/train_onset_ar.py --config configs/ar/tide_overfit.json` |
| **Verify** | `python scripts/eval_ar_onset_offline.py --config configs/ar/tide_overfit.json --ar_decode` |
| **Promote** | `python scripts/graduate_ar_tide_overfit.py --config … --model-path … --version-ref …` |
| **Artifacts** | `models_wsl/ar/tide_overfit/`, `callbacks/ar/tide_overfit/` |

**Current leader:** graduated from **v8** (iter175 scratch) — teacher and free-run **634/634** ordered @ 20 ms vs `target_times`.

Whenever a versioned experiment **beats** the manifest free-run rate, run `graduate_ar_tide_overfit.py` (see [versions/tide_overfit/README.md](versions/tide_overfit/README.md)).

## Versioned experiments (do not delete)

| Directory | Contents |
| --------- | -------- |
| [`versions/tide_overfit/`](versions/tide_overfit/) | v1–v7 frozen recipes + promotion table |
| [`versions/tide_overfit_decode/`](versions/tide_overfit_decode/) | Scheduled-sampling warm-start experiments |
