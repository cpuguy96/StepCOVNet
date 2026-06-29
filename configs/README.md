# Experiment configs

JSON configs for training, eval, and smoke runs. Paths are repo-relative; pass to `--config` on scripts.

## Layout

| Directory                                    | Track                                | Contents                                            |
| -------------------------------------------- | ------------------------------------ | --------------------------------------------------- |
| [`ar/`](ar/)                                 | Autoregressive onset (`onset_ar/`)   | Tide gate, decode gates, perfect-overfit runs       |
| [`ar/decode/`](ar/decode/)                   | AR scheduled-sampling / decode gates | `v2.json`, `perfect.json`, `tide.json`              |
| [`ar/overfit_perfect/`](ar/overfit_perfect/) | AR tide perfect-overfit series       | `base.json`, `run2.json` … `run5.json`              |
| [`dense/`](dense/)                           | Dense frame onset                    | Baseline, `final_data` scoreboard, mel/MERT compare |
| [`event/`](event/)                           | K-query event onset                  | Multi-song baseline, single-song overfit            |
| [`arrow/`](arrow/)                           | Arrow model                          | Baseline, sweep, local dev configs                  |
| [`local/`](local/)                           | Dev / e2e smoke                      | MERT e2e WSL and CPU                                |
| [`overfit_tide/`](overfit_tide/)             | Tide single-song suites              | conv1d / mel / MERT / dense frontends               |
| [`overfit_dense/`](overfit_dense/)           | Dense multi-song overfit             | e.g. 3-song MERT                                    |

## Common entry points

| Goal                          | Config                                      |
| ----------------------------- | ------------------------------------------- |
| AR `gate-tide-overfit`        | `configs/ar/tide.json`                      |
| AR perfect overfit (latest)   | `configs/ar/overfit_perfect/run5.json`      |
| AR `gate-ar-decode` v2        | `configs/ar/decode/v2.json`                 |
| Dense `final_data` scoreboard | `configs/dense/final_data_mert_bilstm.json` |
| Event multi-song baseline     | `configs/event/audio_baseline.json`         |
| Tide overfit suite            | `configs/overfit_tide/mert.json` (etc.)     |

## Onset timing metrics (all tracks)

**Primary (compare models / pass gate):** `timing_match` — sorted predicted vs reference onsets; count ordered pairs with `|pred[i] − ref[i]| ≤ 20 ms`; score = **n_matched / max(n_pred, n_ref)** (e.g. **634/634** on tide overfit when counts match).

| Track | Training / eval name | Offline script |
| ----- | -------------------- | -------------- |
| AR | `val_ordered_onset_match` (alias) | `debug_ar_onset_overfit.py` (`--ar_decode` for free-run) |
| Dense | `val_timing_match` | `eval_dense_onset.py` → `micro_timing_match` |
| Event | `timing_match` in diagnostic JSON | `debug_onset_overfit.py` |

**Checkpoint (AR perfect overfit):** `val_overfit_gate` = `min(val_token_accuracy, val_ordered_onset_match)`.

Hungarian `event_f1` / `val_event_onset_f1` remains **auxiliary** only.

## Migration (2026-06)

Flat `configs/onset_*.json` and `configs/onset_ar_*.json` paths moved into the directories above. Historical `EXPERIMENT_LOG.md` entries keep old paths as written at run time.
