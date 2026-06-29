# Experiment configs

JSON configs for training, eval, and smoke runs. Paths are repo-relative; pass to `--config` on scripts.

## Layout

| Directory                                    | Track                                | Contents                                            |
| -------------------------------------------- | ------------------------------------ | --------------------------------------------------- |
| [`ar/`](ar/)                                 | Autoregressive onset (`onset_ar/`)   | Champion [`tide_overfit.json`](ar/tide_overfit.json); versioned experiments in [`ar/versions/`](ar/versions/) |
| [`dense/`](dense/)                           | Dense frame onset                    | Baseline, `final_data` scoreboard, mel/MERT compare |
| [`event/`](event/)                           | K-query event onset                  | Multi-song baseline, single-song overfit            |
| [`arrow/`](arrow/)                           | Arrow model                          | Baseline, sweep, local dev configs                  |
| [`local/`](local/)                           | Dev / e2e smoke                      | MERT e2e WSL and CPU                                |
| [`overfit_tide/`](overfit_tide/)             | Tide single-song suites              | conv1d / mel / MERT / dense frontends               |
| [`overfit_dense/`](overfit_dense/)           | Dense multi-song overfit             | e.g. 3-song MERT                                    |

## Common entry points

| Goal                          | Config                                      |
| ----------------------------- | ------------------------------------------- |
| AR tide overfit (champion)    | `configs/ar/tide_overfit.json` → `models_wsl/ar/tide_overfit/` |
| AR tide overfit experiments | `configs/ar/versions/tide_overfit/vN.json` (frozen history) |
| Dense `final_data` scoreboard | `configs/dense/final_data_mert_bilstm.json` |
| Event multi-song baseline     | `configs/event/audio_baseline.json`         |
| Tide overfit suite            | `configs/overfit_tide/mert.json` (etc.)     |

## Onset timing metrics

See **[docs/research/ONSET_METRICS.md](../docs/research/ONSET_METRICS.md)** for the full definition (`timing_match`, gates, Hungarian F1 aux, per-track log names).

**Summary:** primary score = ordered match @ 20 ms, rate = `n_matched / max(n_pred, n_ref)`; tide overfit pass = **1.0** (634/634 when counts match).

## Migration (2026-06)

Flat `configs/onset_*.json` and `configs/onset_ar_*.json` paths moved into the directories above. AR artifact dirs renamed under `models_wsl/ar/<gate_slug>/` — see [`ar/README.md`](ar/README.md). Historical `EXPERIMENT_LOG.md` entries keep old paths as written at run time.
