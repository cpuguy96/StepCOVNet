# Decisions checklist — onset event pipeline

**Purpose:** Items to decide before / during ablation runs. Mark status as we go: `open` · `decided` · `deferred`.

**Timestamps:** Research log entries use `YYYY-MM-DD HH:MM:SS` from the system clock at write time (see [research-logging.mdc](../../.cursor/rules/research-logging.mdc) and [research-session-workflow skill](../../.cursor/skills/research-session-workflow/SKILL.md)).

**Pipeline:** audio → **pre** → **model** → raw outputs → **post** → final list → **metrics** → training feedback

**Related:** [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md) · [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) · [planning doc](../onset_output_targets_planning.md)

---

## A. Metric contract (what we optimize and report)

| #   | Decision                                 | Options / notes                                                                    | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --- | ---------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1  | **Primary val metric for checkpointing** | Frame F1 during train; canonical event F1 post-hoc (`eval_dense_onset.py`)         | **decided** – dense: in-train save on **`val_onset_f1_score`** (frame @ 0.5); auto post-hoc event-F1 ckpt+threshold export opt-in via `RunConfig.post_hoc_event_f1_export` ([NOTE-20260610-01](DISCUSSION_NOTES.md#note-20260610-01-auto-post-hoc-event-f1-export-implemented)); manual `sweep_val_onset_ckpts.py` + POST sweep otherwise (EXP-11/12, NOTE-09/11); `DenseValEventF1Callback` disabled ([NOTE-20260608-01](DISCUSSION_NOTES.md#note-20260608-01-disable-dense-val-event-f1-callback)) |
| A2  | **Match tolerance**                      | 20 ms default (`tolerance_sec=0.02`) — keep or sweep?                              | **decided** — keep 20 ms for v1 comparisons                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| A3  | **Confidence threshold**                 | 0.5 default — sweep on best checkpoint?                                            | **decided** — **always** val-sweep POST per checkpoint; Gaussian dense ~**0.35** (EXP-07/12); binary ~0.20 (EXP-08-01); never report default 0.05 alone                                                                                                                                                                                                                                                                                                                                              |
| A4  | **Min onset gap (post)**                 | 50 ms default — matches inference + `event_onset_f1_mingap`                        | **decided** — implemented in metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| A5  | **Train vs eval matching**               | Ordered (legacy) vs Hungarian L1 in training loss (`assign_onset_pairs_l1`)        | **decided** — Hungarian L1 in `losses.py` (EXP-08)                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| A6  | **Dense baseline metric parity**         | Same event F1 on extracted peaks vs frame F1 — document both when comparing tracks | **decided** — checkpoint on `val_dense_event_onset_f1`; frame F1 diagnostic; report via `eval_dense_onset.py`                                                                                                                                                                                                                                                                                                                                                                                        |

---

## B. Pre-processing (input to core model)

| #   | Decision                         | Options / notes                                                            | Status                                                            |
| --- | -------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| B1  | **Frontend for event model v1**  | Raw Conv1D vs cached mel vs cached MERT (`preprocess.py` + model frontend) | **decided** — EXP-20260606-07: MERT > mel >> conv1d on tide smoke |
| B2  | **Cache vs on-the-fly features** | MERT/mel `.npy` cache (fast) vs STFT in Keras graph (e2e)                  | **open**                                                          |
| B3  | **Audio I/O**                    | 44.1 kHz mono, peak norm, 300 s cap — keep?                                | **decided** — matches rest of repo                                |
| B4  | **Augmentation**                 | `apply_audio_augment` off for initial ablations?                           | **open** — recommend off until baseline works                     |

---

## C. Core model (middle)

| #   | Decision               | Options / notes                                                                             | Status                                       |
| --- | ---------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------- |
| C1  | **Output formulation** | K query slots (current) vs dense frames vs seq2seq                                          | **open** — stay on query slots for ablations |
| C2  | **`num_queries` K**    | 1024 vs tied to `n_max_onsets` — enforce `K >= max steps`                                   | **decided** — 1024 in baseline config        |
| C3  | **Time head**          | Learnable deltas + uniform grid (normal); GT refs only when `pipeline_check_shortcuts=true` | **decided**                                  |
| C4  | **Encoder capacity**   | Current U-Net + 2 decoder layers — scale up only after pre ablation                         | **deferred**                                 |
| C5  | **Loss weights**       | `lambda_cls`, `lambda_time` (tide overfit used 20 for time)                                 | **open** for full val train                  |

---

## D. Post-processing (raw outputs → final list)

| #   | Decision              | Options / notes                                            | Status                                          |
| --- | --------------------- | ---------------------------------------------------------- | ----------------------------------------------- |
| D1  | **Order of ops**      | conf threshold → min-gap → Hungarian (metrics mingap path) | **decided**                                     |
| D2  | **Min-gap tie-break** | Keep earlier time (current) vs higher confidence           | **decided** — earlier time (matches dense path) |
| D3  | **Extra NMS / dedup** | Needed beyond min-gap for chart clusters?                  | **open** — measure on val                       |
| D4  | **Count derivation**  | From filtered confidences only — no count head             | **decided**                                     |

---

## E. Training protocol

| #   | Decision                | Options / notes                                                      | Status                                                             |
| --- | ----------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| E1  | **First serious run**   | Tide single-song overfit smoke, no shortcuts, all frontends          | **decided** — `configs/overfit_tide/`, `run_overfit_tide_suite.py` |
| E2  | **Epochs / LR**         | 300 ep cap overfit, Adam 5e-3 / 2e-3 normal                          | **decided** for tide smoke                                         |
| E3  | **Overfit gate**        | Real overfit = normal training path; shortcuts = pipeline check only | **decided**                                                        |
| E4  | **WSL GPU**             | All training/eval on WSL venv                                        | **decided**                                                        |
| E5  | **What to log per run** | loss, `event_onset_f1`, `event_onset_f1_mingap`, TP/FP/FN both paths | **decided** — metrics in trainer                                   |

---

## F. Comparison / paper direction

| #   | Decision                        | Options / notes                                               | Status                                           |
| --- | ------------------------------- | ------------------------------------------------------------- | ------------------------------------------------ |
| F1  | **Fair feature ablation table** | Same event head + metric; swap pre only (A–D in planning doc) | **partial** — tide suite (EXP-07); val scale TBD |
| F2  | **Success bar on val**          | Target F1 before mel-in-graph / seq2seq — TBD                 | **open**                                         |
| F3  | **Ship path**                   | Dense MERT baseline vs event list for downstream generator    | **open**                                         |

---

## G. Dataset prep (`final_data`)

| #   | Decision                         | Options / notes                                              | Status                                                                 |
| --- | -------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| G1  | **Prep output layout**           | Nested `{bundle}/{id}/` + multi-chart JSON                 | **decided** — [DATASET_PREP_PIPELINE.md](DATASET_PREP_PIPELINE.md) §1  |
| G2  | **Training loader (P9)**         | `chart_index` per row; `.chart.json` primary                 | **decided** — `training_loader.py`, `pairing.list_training_samples`    |
| G3  | **Train/val split (P8)**         | `training_index.json` flat manifest                          | **open** — blocks multi-song val on `final_data`                       |
| G4  | **Step cap**                     | 2048 steps per chart at export and load                      | **decided** — all 1942 local rows pass cap (EXP-20260622-01)           |
| G5  | **Legacy `.txt` during migration** | Fallback `chart_index=0` for `data/v2`                     | **decided** — until P8 + full migration                                |

---

## Suggested order of work

### Dataset prep

1. **P8 — `training_index.json` + train/val split** — next gate for multi-song training on `final_data`
2. **First `final_data` training run** — dense or event onset after P8

### Onset research (paused pending P8 or explicit waive)

1. ~~**A5 — Hungarian train loss**~~ — done → [EXP-20260606-08](EXPERIMENT_LOG.md#experiment-index)
2. ~~**Debug conv1d 0% F1**~~ — root cause in [NOTE-20260606-13](DISCUSSION_NOTES.md#note-20260606-13-conv1d-zero-f1--confidence-collapse-from-ordered-training)
3. ~~**Compare EXP-08 vs EXP-07**~~ — Hungarian train: conv1d 0%→27%; all frontends ~27–29% ([EXP-20260606-08](EXPERIMENT_LOG.md#experiment-index))
4. ~~**100 ep tide suite**~~ — no overfit → EXP-09
5. ~~**Threshold / loss / arch ablations**~~ — none break plateau → EXP-10
6. ~~**Tide bisection (diagnose + half-cheat)**~~ — formulation ceiling, not bug → EXP-11
7. **Dense MERT tide overfit** — formulation control experiment
8. **Formulation prototype** — dense frames or seq2seq on event metric
9. Multi-song val — only after formulation gate or explicit waive

Update this file when a row moves to **decided**; link the deciding `EXP-…` or `NOTE-…`.
