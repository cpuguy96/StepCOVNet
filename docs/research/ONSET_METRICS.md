# Onset evaluation metrics

**Purpose:** Single reference for how StepCOVNet scores onset **timing** across AR, dense, and event tracks. Implementation: [`src/stepcovnet/timing_match.py`](../../src/stepcovnet/timing_match.py).

**Where metrics appear:**

| Doc | Scope |
| --- | ----- |
| [ONSET_METRICS_NAMING.md](ONSET_METRICS_NAMING.md) | **Naming** — `gate_*`, `timing_match_*`, `aux_*` tiers |
| [ONSET_METRICS_TRAINING.md](ONSET_METRICS_TRAINING.md) | Keras logs each epoch (`val_*`, losses, gates) |
| [ONSET_METRICS_EVALUATION.md](ONSET_METRICS_EVALUATION.md) | Offline scripts on saved checkpoints |

**Related:** [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) · [AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md) · [configs/README.md](../../configs/README.md) · [DISCUSSION_NOTES.md § NOTE-20260628-03](DISCUSSION_NOTES.md#note-20260628-03-tide-overfit-primary-metric--ordered-onset-match)

---

## Why ordered timing match

Chart onsets are an **ordered sequence in time**. A model that finds the right multiset of times but permutes two steps is wrong for chart reproduction and generation.

**Primary metric:** compare the `i`th earliest predicted onset to the `i`th earliest reference onset (after export and sort). No Hungarian reassignment.

**Auxiliary metric:** Hungarian event F1 still logs TP/FP/FN for legacy comparisons and multi-song val, but can hide per-rank timing errors (e.g. F1 **1.0** with **633/634** ordered @ 20 ms on tide).

---

## Definition — `timing_match`

Every track exports a **sorted** list of predicted onset times in seconds (`pred_times`) and compares to sorted reference times (`ref_times`).

| Symbol      | Meaning                                                                                |
| ----------- | -------------------------------------------------------------------------------------- |
| `n_ref`     | Number of reference onsets                                                             |
| `n_pred`    | Number of predicted onsets after POST (peak-pick, slot filter, AR decode, …)           |
| `n_matched` | Count of indices `i < min(n_pred, n_ref)` where `\|pred[i] − ref[i]\| ≤ tolerance_sec` |
| `n_denom`   | `max(n_pred, n_ref)`                                                                   |
| **rate**    | `n_matched / n_denom` (0 if `n_denom == 0`)                                            |

**Default tolerance:** `tolerance_sec = 0.02` (**20 ms**). Fixed across tracks for comparisons unless an experiment explicitly sweeps it.

### Examples (tide has `n_ref = 634`)

| Situation                             | `n_matched` | `n_pred` | `n_ref` | Rate    | Pass @ 1.0? |
| ------------------------------------- | ----------- | -------- | ------- | ------- | ----------- |
| All pairs within 20 ms, correct count | 634         | 634      | 634     | 1.0     | yes         |
| One pair off by >20 ms                | 633         | 634      | 634     | 633/634 | no          |
| First 634 perfect, **one extra** peak | 634         | 635      | 634     | 634/635 | no          |
| 619 correct prefixes, stopped early   | 619         | 619      | 634     | 619/634 | no          |

**Tide overfit pass bar:** rate **1.0** on teacher-fed and free-run AR decode vs **`target_times`** — equivalently **634/634** when `n_pred == n_ref == 634`. Log **`chart_ordered_onset_match`** (raw chart seconds) and Hungarian F1 as aux.

### Training reference vs raw chart (AR)

| Reference | Source | Role |
| --------- | ------ | ---- |
| **`target_times`** | `decode_pointer_residual_to_times` on teacher patch+residual | **Primary** — matches training loss and hop-grid output space |
| **`gt_times` / chart** | Clipped `tide.txt` seconds | **Aux** — annotation fidelity; can differ by up to ~one hop from `target_times` |

Evaluating free-run only against raw chart while training supervises `target_times` creates false failures when residual error is near tolerance ([NOTE-20260630-01](DISCUSSION_NOTES.md#note-20260630-01-ar-free-run-primary-vs-target_times)).

### Micro aggregation (multi-song val)

Across songs, dense val reports:

```text
micro_timing_match = sum(n_matched) / max(sum(n_pred), sum(n_ref))
```

Per-song rates are also available in JSON (`timing_match_rate` per song); micro vs mean-per-song can diverge when song lengths differ.

---

## Pipeline: export then score

```mermaid
flowchart LR
  M[MODEL raw outputs] --> P[POST export]
  P --> S[Sorted pred_times]
  G[Sorted ref_times] --> T[timing_match]
  S --> T
  T --> R[rate + counts]
```

| Track             | POST export                                                                | Reference times                                |
| ----------------- | -------------------------------------------------------------------------- | ---------------------------------------------- |
| **AR (teacher)**  | Pointer+residual decode at masked onset steps                              | Training `target_times` at same steps          |
| **AR (free-run)** | Two-pass autoregressive decode → times                                     | Training `target_times` (primary); raw `gt_times` aux |
| **Dense**         | Peak-pick on frame probs (`confidence_threshold`, `min_onset_distance_ms`) | Frame indices → seconds via hop                |
| **Event**         | Filter slots by confidence + min-gap → sort                                | `reference_times_from_mask(gt_times, gt_mask)` |

Order is **enforced at eval** by sorting both sides. AR decode is chronological by construction; dense/event rely on POST sort.

---

## Names by track

| Track           | Canonical name | Training / val log name             | Offline script                                       |
| --------------- | -------------- | ----------------------------------- | ---------------------------------------------------- |
| **AR**          | `timing_match` | `val_ordered_onset_match` (teacher) | `scripts/debug_ar_onset_overfit.py`                  |
| **AR free-run** | `timing_match` | `val_ar_decode_ordered_onset_match` | same + `--ar_decode` (gate default; `--full-diagnostics` for slow extras) |
| **Dense**       | `timing_match` | `val_timing_match` (callback)       | `scripts/eval_dense_onset.py` → `micro_timing_match` |
| **Event**       | `timing_match` | _(diagnostics only today)_          | `scripts/debug_onset_overfit.py`                     |

Backward-compatible JSON keys `ordered_onset_match` still appear in AR debug output alongside `timing_match`.

---

## Composite gates (AR overfit)

| Metric             | Formula                                            | Use                                    |
| ------------------ | -------------------------------------------------- | -------------------------------------- |
| `val_overfit_gate` | `min(val_token_accuracy, val_ordered_onset_match)` | Checkpoint on perfect-overfit configs  |
| Offline PASS       | `rate >= 1.0` (and `n_denom > 0`)                  | Human-readable stderr in debug scripts |

Free-run perfect-overfit also requires **`val_ar_decode_ordered_onset_match == 1.0`** (offline `--ar_decode`, two-pass timing). Token accuracy alone is insufficient.

---

## Auxiliary — Hungarian event F1

**Module:** `onset_events/matching.py`, `onset_events/metrics.py` (dense peak-pick reuses event counting).

| Setting        | Default                                      |
| -------------- | -------------------------------------------- |
| Matching       | Hungarian one-to-one, minimize total \|Δt\|  |
| Tolerance gate | Pair counts only if \|Δt\| ≤ `tolerance_sec` |
| Confidence     | TP requires `confidence ≥ threshold`         |

| Log name                 | Track                                   |
| ------------------------ | --------------------------------------- |
| `val_event_onset_f1`     | AR teacher, event training              |
| `val_ar_decode_event_f1` | AR free-run callback                    |
| `micro_event_f1`         | Dense val eval script                   |
| `event_onset_f1_mingap`  | Event path with 50 ms POST before match |

**When to use:** multi-song val scoreboard, detection vs timing decomposition (TP/FP/FN), historical EXP comparisons.

**When not to use as primary:** single-song tide overfit — ordered timing match is the pass/fail bar ([NOTE-20260628-03](DISCUSSION_NOTES.md#note-20260628-03-tide-overfit-primary-metric--ordered-onset-match)).

### Hungarian F1 has a high chance floor — always report it

**Module:** `stepcovnet/onset_null_baseline.py` · **Tests:** `tests/onset_null_baseline_test.py`

Hungarian matching is order-free, so a prediction set only has to land *near* some ground-truth onset. On dense charts that is easy by accident: the AR ladder val set averages **5.52** onsets/sec (mean inter-onset interval **181 ms**), so a ±20 ms window covers ~22% of the timeline. Measured floors on that set, at a prediction count matched to the model's:

| Audio-blind predictor | F1 @ `pred/GT` = 1.0 | F1 @ 0.90 | Ordered `timing_match` |
| --------------------- | -------------------- | --------- | ---------------------- |
| Uniform over duration | 0.225 | 0.217 | 0.003 |
| Metronome over GT support | 0.275 | 0.261 | 0.003 |
| Shuffled GT intervals | 0.336 | 0.313 | 0.013 |

None of these hear the audio. A raw Hungarian F1 below ~0.34 on a dense multi-song val set is therefore **not evidence of skill**, and a change that only moves the predicted onset *count* moves F1 along this curve for free — which is how the AR ladder produced a year of unreadable numbers ([EXP-20260804-03](EXPERIMENT_LOG.md#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke)).

Ordered `timing_match` has a floor of ≈ 0 because it also requires the *k*-th prediction to align with the *k*-th reference, which is why it is the primary metric.

```python
from stepcovnet import onset_null_baseline

counts = onset_null_baseline.null_counts_for_song(
    gt_times, duration_sec=duration, n_pred=n_pred_model, tolerance_sec=0.02,
)
floors = onset_null_baseline.aggregate_null_counts([counts])
kind, floor = onset_null_baseline.strongest_null(floors)
skill = onset_null_baseline.skill_over_null(model_f1, floor)  # <= 0 means no skill
```

`scripts/debug_ar_onset_overfit.py` computes this automatically for teacher and free-run blocks: stderr shows `Null F1 @ matched count` and `Skill over strongest null`, and the JSON carries `null_baseline`.

---

## Choosing a metric for an experiment

| Goal                                    | Primary                                    | Also log                                         |
| --------------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| Tide / single-chart overfit             | `timing_match` vs **`target_times`** **1.0** | `chart_ordered_onset_match`, Hungarian F1, token acc |
| AR `gate-ar-decode`                     | Free-run `timing_match` vs **`target_times`** | Teacher `timing_match`, chart aux, decode length, EOS |
| Multi-song dense val                    | `micro_timing_match` @ **fixed** threshold | `micro_event_f1`, TP/FP/FN                       |
| Event K-slot research (plateau ~30% F1) | `event_onset_f1` (still training default)  | `timing_match` in diagnostics when debugging     |

**Null floor caveat:** never report a Hungarian F1 on a dense multi-song val set without its audio-blind floor at the same prediction count — see § *Hungarian F1 has a high chance floor*. For multi-song AR, select checkpoints on `val_timing_match_teacher`, not `val_aux_f1_hungarian`.

**Val model selection caveat:** `timing_match` on dense/event depends on POST (threshold, min-gap). Compare checkpoints only under the **same** export policy, or sweep threshold once and hold it fixed across runs.

**Train vs eval gap (event):** training loss still uses Hungarian L1 assignment on K slots; eval `timing_match` uses sorted export lists. That is acceptable for multi-song F1 research; for event overfit on one chart, consider ordered assignment (`assign_onset_pairs_ordered_numpy`) in a follow-up experiment.

---

## API quick reference

```python
from stepcovnet import timing_match

report = timing_match.timing_match_report(
    pred_times, ref_times, tolerance_sec=0.02,
)
# report: n_matched, n_pred, n_ref, n_denom, rate, tolerance_sec

rate = timing_match.timing_match_rate_numpy(pred_times, ref_times, tolerance_sec=0.02)

ref = timing_match.reference_times_from_mask(gt_times[0], gt_mask[0])
```

**Tests:** `tests/timing_match_test.py` · AR trainers: `tests/onset_ar/trainers_test.py` · dense: `tests/dense_overfit_eval_test.py`

---

## Changelog

| Date       | Change                                                                                                                                                                      |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-04 | Added `onset_null_baseline`: Hungarian F1 @ 20 ms has a **0.225–0.336** chance floor on dense charts, so every reported F1 now carries its audio-blind floor and skill ([EXP-20260804-03](EXPERIMENT_LOG.md#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke)). `timing_match_teacher` now published on multi-song AR runs |
| 2026-06-30 | Free-run primary ordered match vs **`target_times`**; raw chart as **`chart_ordered_onset_match`** aux ([NOTE-20260630-01](DISCUSSION_NOTES.md#note-20260630-01-ar-free-run-primary-vs-target_times)) |
| 2026-06-28 | Primary tide metric switched from Hungarian F1 to ordered match ([NOTE-20260628-03](DISCUSSION_NOTES.md#note-20260628-03-tide-overfit-primary-metric--ordered-onset-match)) |
| 2026-06-28 | Unified `timing_match.py` across AR / dense / event; rate denominator `max(n_pred, n_ref)` penalizes extra predictions                                                      |
