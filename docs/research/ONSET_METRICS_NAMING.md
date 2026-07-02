# Onset metric naming convention

**Purpose:** One vocabulary for training logs, eval JSON, and checkpoints. Tier prefixes show what matters for decisions vs debugging.

**Implementation:** [`src/stepcovnet/onset_metric_names.py`](../../src/stepcovnet/onset_metric_names.py)

**Where metrics appear:** [ONSET_METRICS_TRAINING.md](ONSET_METRICS_TRAINING.md) · [ONSET_METRICS_EVALUATION.md](ONSET_METRICS_EVALUATION.md)

---

## Tier prefixes

| Prefix | Tier | Use for decisions? | Examples |
| ------ | ---- | ------------------ | -------- |
| `gate_*` | **Gate** | Yes — checkpoint / early stop | `gate_teacher` |
| `timing_match_*` | **Primary** | Yes — pass/fail bar | `timing_match_teacher`, `timing_match_ar_decode` |
| `token_accuracy` | **Support** | Yes — component of teacher gate | `token_accuracy` |
| `aux_*` | **Auxiliary** | No — legacy / decomposition | `aux_f1_hungarian`, `aux_timing_match_chart` |
| `loss_*` | **Loss** | No — optimization only | `token_loss`, `residual_loss` |
| `diag_*` | **Debug** | No — offline diagnosis | `worst_onsets`, `abs_error_ms` |

Keras validation logs add a `val_` prefix (e.g. `val_gate_teacher`).

---

## Canonical names (prefer these)

### AR tide overfit

| Canonical | Meaning | Pass bar (tide) |
| --------- | ------- | ----------------- |
| **`timing_match_teacher`** | Ordered timing @ 20 ms vs `target_times`, teacher-fed | **634/634** |
| **`timing_match_ar_decode`** | Same metric, free-run two-pass decode | **634/634** |
| **`gate_teacher`** | `min(token_accuracy, timing_match_teacher)` | **1.0** |
| `token_accuracy` | Teacher-fed token CE accuracy | **1.0** (for gate) |
| `aux_f1_hungarian` | Hungarian event F1 vs raw chart | informational |
| `aux_timing_match_chart` | Ordered match vs raw `gt_times` | informational |

### Dense multi-song eval

| Canonical | Meaning |
| --------- | ------- |
| **`timing_match_micro`** | Micro-averaged ordered timing across val songs |
| `aux_f1_hungarian` | Micro Hungarian event F1 (`micro_event_f1` in JSON today) |

### Event training

| Canonical | Meaning |
| --------- | ------- |
| `aux_f1_hungarian` | Default checkpoint F1 (no min-gap) |
| `aux_f1_hungarian_mingap` | F1 after 50 ms POST |

---

## Legacy → canonical map

Old names still appear in logs and configs during transition. Code dual-publishes both.

| Legacy (still accepted) | Canonical (preferred) |
| ----------------------- | --------------------- |
| `val_overfit_gate` | `val_gate_teacher` |
| `val_ordered_onset_match` | `val_timing_match_teacher` |
| `val_event_onset_f1` | `val_aux_f1_hungarian` |
| `val_ar_decode_ordered_onset_match` | `val_timing_match_ar_decode` |
| `val_ar_decode_event_f1` | `val_aux_f1_hungarian_ar_decode` |
| `ordered_onset_match` (eval JSON) | `timing_match_teacher` |
| `chart_ordered_onset_match` | `aux_timing_match_chart` |
| `event_f1` (eval JSON) | `aux_f1_hungarian` |
| `micro_timing_match` (dense JSON) | `timing_match_micro` |
| `micro_event_f1` | `aux_f1_hungarian` (micro scope) |

`checkpoint_metric` in configs accepts **either** legacy or canonical; training resolves to the Keras monitor name.

---

## What to watch (cheat sheet)

### During AR tide training

```text
val_gate_teacher          ← checkpoint / early stop
val_timing_match_teacher  ← primary timing
val_token_accuracy        ← token support
val_aux_f1_hungarian      ← ignore for pass/fail
```

### After training (offline eval)

```text
metrics_by_tier.primary.timing_match_teacher
metrics_by_tier.primary.timing_match_ar_decode   ← true PASS requires 634/634
metrics_by_tier.aux.*                            ← debugging only
```

Eval JSON includes both flat legacy keys and `metrics_by_tier` grouping.

---

## Config examples

```json
{
  "run": {
    "checkpoint_metric": "val_gate_teacher",
    "perfect_overfit_early_stop": true,
    "perfect_overfit_min_score": 0.9999
  }
}
```

Legacy equivalent: `"checkpoint_metric": "val_overfit_gate"` — still works.

---

## Related

- [ONSET_METRICS.md](ONSET_METRICS.md) — definitions and tolerance
- [ONSET_METRICS_TRAINING.md](ONSET_METRICS_TRAINING.md)
- [ONSET_METRICS_EVALUATION.md](ONSET_METRICS_EVALUATION.md)
