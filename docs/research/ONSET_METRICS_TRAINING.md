# Onset metrics during training

**Purpose:** What Keras logs each epoch while `fit()` runs — losses, validation metrics, and checkpoint gates. For metric definitions and pass bars, see [ONSET_METRICS.md](ONSET_METRICS.md). For offline checkpoint scoring, see [ONSET_METRICS_EVALUATION.md](ONSET_METRICS_EVALUATION.md).

**Implementation:** AR [`src/stepcovnet/onset_ar/trainers.py`](../../src/stepcovnet/onset_ar/trainers.py) · event [`src/stepcovnet/onset_events/trainers.py`](../../src/stepcovnet/onset_events/trainers.py) · dense [`src/stepcovnet/trainers.py`](../../src/stepcovnet/trainers.py)

---

## How to read training logs

Every metric below is prefixed with `val_` on the validation split.

| Tier | Meaning |
| ---- | ------- |
| **Critical** | Use for checkpointing, early stop, or “is this run working?” |
| **Support** | Explains why a critical metric is stuck |
| **Debug** | Loss decomposition and legacy/aux scores — log for diagnosis only |

**Naming:** See [ONSET_METRICS_NAMING.md](ONSET_METRICS_NAMING.md) for tier prefixes (`gate_*`, `timing_match_*`, `aux_*`).

---

## AR onset (`train_onset_ar.py`)

Trainer: `ArOnsetTrainingModel` in `onset_ar/trainers.py`.

### Critical (tide overfit)

| Log name | What it is | Decision use |
| -------- | ---------- | ------------ |
| **`val_gate_teacher`** | `min(val_token_accuracy, val_timing_match_teacher)` | Default checkpoint metric (legacy: `val_overfit_gate`) |
| **`val_timing_match_teacher`** | Teacher-fed **`timing_match`** vs `target_times` | Primary timing (legacy: `val_ordered_onset_match`) |
| **`val_token_accuracy`** | Fraction of decoder tokens correct (teacher-fed) | Component of `val_overfit_gate`. Catches token collapse (e.g. majority-token plateau ~0.48). |

**Not logged during training (champion recipe):** free-run `timing_match`. Tide config sets `ar_decode_val_every_n_epochs: 0`. Run [offline eval](ONSET_METRICS_EVALUATION.md#ar-onset-debug_ar_onset_overfitpy) after training for the true pass bar.

**Tide pass during training:** `val_overfit_gate → 1.0` is **necessary, not sufficient**. Final PASS still requires free-run `timing_match == 1.0` offline.

### Support

| Log name | What it is | When it matters |
| -------- | ---------- | --------------- |
| `val_aux_f1_hungarian` | Hungarian event F1 (teacher-fed vs raw `gt_times`) | Legacy: `val_event_onset_f1`. Aux only. |
| `token_accuracy` | Train-split token accuracy | Same as val counterpart; useful while loss is still dropping. |

### Debug — loss decomposition

Logged on train and val (`val_` prefix on val). These are **optimization targets**, not pass/fail metrics.

| Log name | Source |
| -------- | ------ |
| `loss` | Weighted sum of all loss terms |
| `token_loss` | Causal CE on `delta_bucketed` tokens (+ EOS) |
| `pointer_loss` | Cross-entropy on patch pointer |
| `time_loss` | Auxiliary time loss (ramped via `lambda_time_ramp_epochs`) |
| `residual_loss` | L1 on residual seconds (`lambda_residual`) |
| `incremental_consistency_loss` | Pointer consistency across decode steps (`lambda_incremental_consistency`) |

### Callbacks (not separate log lines)

| Callback | Effect |
| -------- | ------ |
| `OverfitGateCallback` | Writes `val_overfit_gate`; optional early stop when `perfect_overfit_early_stop: true` and gate ≥ `perfect_overfit_min_score` for `perfect_overfit_patience` epochs |
| `LambdaTimeRampCallback` | Ramps `lambda_time` when `lambda_time_ramp_epochs > 0` |
| `ScheduledSamplingRampCallback` | Ramps scheduled-sampling probability when enabled |
| `ModelCheckpoint` | Saves best weights on `checkpoint_metric` (tide: `val_overfit_gate`) |

### Example epoch line (tide)

```text
val_token_accuracy: 1.0
val_ordered_onset_match: 0.9890
val_overfit_gate: 0.9890
val_event_onset_f1: 0.9858
```

Read as: tokens perfect; **7** teacher-fed onsets off by >20 ms; gate capped by timing, not tokens.

---

## Event / K-query onset (`train_onset_event.py`)

Trainer: `OnsetEventTrainingModel` in `onset_events/trainers.py`.

### Critical

| Log name | What it is | Decision use |
| -------- | ---------- | ------------ |
| **`val_event_onset_f1`** | Hungarian event F1 @ 20 ms (no min-gap POST) | Default `checkpoint_metric` for event configs |
| `val_loss` | Combined cls + time loss | Fallback checkpoint when F1 is flat |

### Support

| Log name | What it is |
| -------- | ---------- |
| `val_event_onset_f1_mingap` | Same F1 after 50 ms min-gap POST (inference-style filtering) |

### Debug

| Log name | What it is |
| -------- | ---------- |
| `loss` | Training loss |
| `event_onset_f1` | Train-split F1 |

**Note:** Event training loss uses Hungarian assignment; eval `timing_match` is **not** logged in-loop today. Use offline scripts if you need ordered timing on a checkpoint.

---

## Dense / frame onset (`train_onset_dense.py`)

Trainer: compiled U-Net in `stepcovnet/trainers.py`.

### Critical

| Log name | What it is | Decision use |
| -------- | ---------- | ------------ |
| **`val_onset_f1_score`** | Frame-level onset F1 (`OnsetF1Metric`, ±2 frames @ threshold 0.5) | Default dense checkpoint monitor |
| `val_loss` | Training loss | Early stopping / debugging |

### Support (frame quality)

| Log name | What it is |
| -------- | ---------- |
| `val_acc` | Binary frame accuracy |
| `val_prec` / `val_rec` | Frame precision / recall |
| `val_auc` / `val_pr_auc` | ROC / PR AUC on frames |

### Not in training logs

Ordered **`timing_match`** and peak-pick **event F1** are computed **only offline** via [`scripts/eval_dense_onset.py`](../../scripts/eval_dense_onset.py). Dense training optimizes frame labels; export POST (threshold, min-gap) affects eval scores.

---

## Quick reference — what to watch

| Track | Watch during training | Do not treat as final pass |
| ----- | --------------------- | -------------------------- |
| **AR tide overfit** | `val_overfit_gate`, `val_ordered_onset_match`, `val_token_accuracy` | `val_event_onset_f1` alone; any metric without offline `--ar_decode` |
| **AR multi-song** | `val_event_onset_f1` (default checkpoint) | — |
| **Event** | `val_event_onset_f1` | — |
| **Dense** | `val_onset_f1_score`, `val_loss` | Frame F1 ≠ chart `timing_match` — run `eval_dense_onset.py` |

---

## Related

- [ONSET_METRICS.md](ONSET_METRICS.md) — definitions, tolerance, pass bars
- [ONSET_METRICS_EVALUATION.md](ONSET_METRICS_EVALUATION.md) — offline eval scripts and JSON fields
- [AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md) — AR gates and champion config
